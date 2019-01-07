/*
 * jpeg_read.c
 *
 * JPEGOBJ = JPEG_READ(FILENAME)
 *
 * Returns JPEGOBJ, a Matlab struct containing the JPEG header information,
 * quantization tables, and the DCT coefficients.
 *
 * This software is based in part on the work of the Independent JPEG Group.
 * In order to compile, you must first build IJG's JPEG Tools code library, 
 * available at ftp://ftp.uu.net/graphics/jpeg/jpegsrc.v6b.tar.gz.
 *
 * Phil Sallee 6/2003
 * 
 * Copyright (c) 2003 The Regents of the University of California. 
 * All Rights Reserved. 
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for educational, research and non-profit purposes,
 * without fee, and without a written agreement is hereby granted,
 * provided that the above copyright notice, this paragraph and the
 * following three paragraphs appear in all copies.
 * 
 * Permission to incorporate this software into commercial products may
 * be obtained by contacting the University of California.  Contact Jo Clare
 * Peterman, University of California, 428 Mrak Hall, Davis, CA, 95616.
 * 
 * This software program and documentation are copyrighted by The Regents
 * of the University of California. The software program and
 * documentation are supplied "as is", without any accompanying services
 * from The Regents. The Regents does not warrant that the operation of
 * the program will be uninterrupted or error-free. The end-user
 * understands that the program was developed for research purposes and
 * is advised not to rely exclusively on the program for any reason.
 * 
 * IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY
 * FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
 * INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
 * ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF
 * CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS"
 * BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE
 * MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <jerror.h>
#include <jpeglib.h>
#include <setjmp.h>
#include "mex.h"


/* We need to create our own error handler so that we can override the 
 * default handler in case a fatal error occurs.  The standard error_exit
 * method calls exit() which doesn't clean things up properly and also 
 * exits Matlab.  This is described in the example.c routine provided in
 * the IJG's code library.
 */
struct my_error_mgr {
  struct jpeg_error_mgr pub;	/* "public" fields */
  jmp_buf setjmp_buffer;	/* for return to caller */
};


/* The default output_message routine causes a seg fault in Matlab,
 * at least on Windows.  Its generally used to emit warnings, since
 * fatal errors call the error_exit routine, so we emit a Matlab
 * warning instead.  If desired, warnings can be turned off by the
 * user with "warnings off".   -- PAS 10/03
*/
METHODDEF(void)
my_output_message (j_common_ptr cinfo)
{
  char buffer[JMSG_LENGTH_MAX];

  /* Create the message */
  (*cinfo->err->format_message) (cinfo, buffer);

  mexWarnMsgTxt(buffer);
}

typedef struct my_error_mgr * my_error_ptr;

METHODDEF(void)
my_error_exit (j_common_ptr cinfo)
{
  char buffer[JMSG_LENGTH_MAX];

  /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
  my_error_ptr myerr = (my_error_ptr) cinfo->err;

  /* create the message */
  (*cinfo->err->format_message) (cinfo, buffer);
  printf("Error: %s\n",buffer);
  
  /* return control to the setjmp point */
  longjmp(myerr->setjmp_buffer, 1);
}


/* Substitute for mxCreateDoubleScalar, which is not available
 * on earlier versions of Matlab */
mxArray *mxCDS(double val) {
  mxArray *mxtemp;
  double *p;

  mxtemp = mxCreateDoubleMatrix(1, 1, mxREAL);
  p = mxGetPr(mxtemp);
  *p = val;

  return mxtemp;
}


/* mex function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  struct jpeg_decompress_struct cinfo;
  struct my_error_mgr jerr;
  jpeg_component_info *compptr;
  jvirt_barray_ptr *coef_arrays;
  jpeg_saved_marker_ptr marker_ptr;
  FILE *infile;
  JDIMENSION blk_x,blk_y;
  JBLOCKARRAY buffer;
  JCOEFPTR bufptr;
  JQUANT_TBL *quant_ptr;
  JHUFF_TBL *huff_ptr;
  int strlen, c_width, c_height, ci, i, j, n, dims[2];
  char *filename;
  double *mp, *mptop;
  mxChar *mcp;
  mxArray *mxtemp, *mxjpeg_obj, *mxcoef_arrays, *mxcomments;
  mxArray *mxquant_tables, *mxhuff_tables, *mxcomp_info;

  /* field names jpeg_obj Matlab struct */
  const char *jobj_field_names[] = {
    "image_width",          /* image width in pixels */
    "image_height",         /* image height in pixels */
    "image_components",     /* number of image color components */
    "image_color_space",    /* in/out_color_space */
    "jpeg_components",      /* number of JPEG color components */
    "jpeg_color_space",     /* color space of DCT coefficients */
    "comments",             /* COM markers, if any */
    "coef_arrays",          /* DCT arrays for each component */
    "quant_tables",         /* quantization tables */
    "ac_huff_tables",       /* AC huffman encoding tables */
    "dc_huff_tables",       /* DC huffman encoding tables */
    "optimize_coding",      /* flag to optimize huffman tables */
    "comp_info",            /* component info struct array */
    "progressive_mode",     /* is progressive mode */
  };
  const int num_jobj_fields = 14;

  /* field names comp_info struct */
  const char *comp_field_names[] = {
    "component_id",         /* JPEG one byte identifier code */
    "h_samp_factor",        /* horizontal sampling factor */
    "v_samp_factor",        /* vertical sampling factor */
    "quant_tbl_no",         /* quantization table number for component */
    "dc_tbl_no",            /* DC entropy coding table number */
    "ac_tbl_no"             /* AC entropy encoding table number */
  };
  const int num_comp_fields = 6;

  const char *huff_field_names[] = {"counts","symbols"};

  /* check input value */
  if (nrhs != 1) mexErrMsgTxt("One input argument required.");
  if (mxIsChar(prhs[0]) != 1)
    mexErrMsgTxt("Filename must be a string");

  /* check output return jpegobj struct */
  if (nlhs > 1) mexErrMsgTxt("Too many output arguments");

  /* get filename */
  strlen = mxGetM(prhs[0])*mxGetN(prhs[0]) + 1;
  filename = mxCalloc(strlen, sizeof(char));
  mxGetString(prhs[0],filename,strlen);

  /* open file */
  if ((infile = fopen(filename, "rb")) == NULL)
    mexErrMsgTxt("Can't open file");

  /* set up the normal JPEG error routines, then override error_exit. */
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = my_error_exit;
  jerr.pub.output_message = my_output_message;

  /* establish the setjmp return context for my_error_exit to use. */
  if (setjmp(jerr.setjmp_buffer)) {
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    mexErrMsgTxt("Error reading file");
  }

  /* initialize JPEG decompression object */
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);

  /* save contents of markers */
  jpeg_save_markers(&cinfo, JPEG_COM, 0xFFFF);

  /* read header and coefficients */
  jpeg_read_header(&cinfo, TRUE);

  /* create Matlab jpegobj struct */
  mxjpeg_obj = mxCreateStructMatrix(1,1,num_jobj_fields,jobj_field_names);

  /* for some reason out_color_components isn't being set by
     jpeg_read_header, so we will infer it from out_color_space: */
  switch (cinfo.out_color_space) {
    case JCS_GRAYSCALE:
      cinfo.out_color_components = 1;
      break;
    case JCS_RGB:
      cinfo.out_color_components = 3;
      break;
    case JCS_YCbCr:
      cinfo.out_color_components = 3;
      break;
    case JCS_CMYK:
      cinfo.out_color_components = 4;
      break;
    case JCS_YCCK:
      cinfo.out_color_components = 4;
      break;
  }

  /* copy header information */
  mxSetField(mxjpeg_obj,0,"image_width",
    mxCDS(cinfo.image_width));
  mxSetField(mxjpeg_obj,0,"image_height",
    mxCDS(cinfo.image_height));
  mxSetField(mxjpeg_obj,0,"image_color_space",
    mxCDS(cinfo.out_color_space));
  mxSetField(mxjpeg_obj,0,"image_components",
    mxCDS(cinfo.out_color_components));
  mxSetField(mxjpeg_obj,0,"jpeg_color_space",
    mxCDS(cinfo.jpeg_color_space));
  mxSetField(mxjpeg_obj,0,"jpeg_components",
    mxCDS(cinfo.num_components));
  mxSetField(mxjpeg_obj,0,"progressive_mode",
    mxCDS(cinfo.progressive_mode));

  /* set optimize_coding flag for jpeg_write() */
  mxSetField(mxjpeg_obj,0,"optimize_coding",mxCDS(FALSE));

  /* copy component information */
  mxcomp_info = mxCreateStructMatrix(1,cinfo.num_components,
    num_comp_fields,comp_field_names);
  mxSetField(mxjpeg_obj,0,"comp_info",mxcomp_info);
  for (ci = 0; ci < cinfo.num_components; ci++) {
    mxSetField(mxcomp_info,ci,"component_id",
      mxCDS(cinfo.comp_info[ci].component_id));
    mxSetField(mxcomp_info,ci,"h_samp_factor",
      mxCDS(cinfo.comp_info[ci].h_samp_factor));
    mxSetField(mxcomp_info,ci,"v_samp_factor",
      mxCDS(cinfo.comp_info[ci].v_samp_factor));
    mxSetField(mxcomp_info,ci,"quant_tbl_no",
      mxCDS(cinfo.comp_info[ci].quant_tbl_no+1));
    mxSetField(mxcomp_info,ci,"ac_tbl_no",
      mxCDS(cinfo.comp_info[ci].ac_tbl_no+1));
    mxSetField(mxcomp_info,ci,"dc_tbl_no",
      mxCDS(cinfo.comp_info[ci].dc_tbl_no+1));
  }

  /* copy markers */
  mxcomments = mxCreateCellMatrix(0,0);
  mxSetField(mxjpeg_obj,0,"comments",mxcomments);
  marker_ptr = cinfo.marker_list;
  while (marker_ptr != NULL) {
    switch (marker_ptr->marker) {
      case JPEG_COM:
         /* this comment index */
         n = mxGetN(mxcomments);

         /* allocate space in cell array for a new comment */
         mxSetPr(mxcomments,mxRealloc(mxGetPr(mxcomments),
                 (n+1)*mxGetElementSize(mxcomments)));
         mxSetM(mxcomments,1);
         mxSetN(mxcomments,n+1);

         /* create new char array to store comment string */
         dims[0] = 1;
         dims[1] = marker_ptr->data_length;
         mxtemp = mxCreateCharArray(2,dims);
         mxSetCell(mxcomments,n,mxtemp);
         mcp = (mxChar *) mxGetPr(mxtemp);

         /* copy comment string to char array */
         for (i = 0; i < (int) marker_ptr->data_length; i++) 
           *mcp++ = (mxChar) marker_ptr->data[i];

         break;
      default:
         break;
    }
    marker_ptr = marker_ptr->next;
  }

  /* copy the quantization tables */
  mxquant_tables = mxCreateCellMatrix(1,NUM_QUANT_TBLS);
  mxSetField(mxjpeg_obj,0,"quant_tables",mxquant_tables);
  mxSetN(mxquant_tables, 0);
  for (n = 0; n < NUM_QUANT_TBLS; n++) {
    if (cinfo.quant_tbl_ptrs[n] != NULL) {
      mxSetN(mxquant_tables, n+1);
      mxtemp = mxCreateDoubleMatrix(DCTSIZE, DCTSIZE, mxREAL);
      mxSetCell(mxquant_tables,n,mxtemp);
      mp = mxGetPr(mxtemp);
      quant_ptr = cinfo.quant_tbl_ptrs[n];
      for (i = 0; i < DCTSIZE; i++) 
        for (j = 0; j < DCTSIZE; j++)
          mp[j*DCTSIZE+i] = (double) quant_ptr->quantval[i*DCTSIZE+j];
    }
  }

  /* copy the AC huffman tables */
  mxhuff_tables = mxCreateStructMatrix(1,NUM_HUFF_TBLS,2,huff_field_names);
  mxSetField(mxjpeg_obj,0,"ac_huff_tables",mxhuff_tables);
  mxSetN(mxhuff_tables, 0);
  for (n = 0; n < NUM_HUFF_TBLS; n++) {
    if (cinfo.ac_huff_tbl_ptrs[n] != NULL) {
      huff_ptr = cinfo.ac_huff_tbl_ptrs[n];
      mxSetN(mxhuff_tables, n+1);

      mxtemp = mxCreateDoubleMatrix(1, 16, mxREAL);
      mxSetField(mxhuff_tables,n,"counts",mxtemp);
      mp = mxGetPr(mxtemp);
      for (i = 1; i <= 16; i++) *mp++ = huff_ptr->bits[i];

      mxtemp = mxCreateDoubleMatrix(1, 256, mxREAL);
      mxSetField(mxhuff_tables,n,"symbols",mxtemp);
      mp = mxGetPr(mxtemp);
      for (i = 0; i < 256; i++) *mp++ = huff_ptr->huffval[i];
    }
  }

  /* copy the DC huffman tables */
  mxhuff_tables = mxCreateStructMatrix(1,NUM_HUFF_TBLS,2,huff_field_names);
  mxSetField(mxjpeg_obj,0,"dc_huff_tables",mxhuff_tables);
  mxSetN(mxhuff_tables, 0);
  for (n = 0; n < NUM_HUFF_TBLS; n++) {
    if (cinfo.dc_huff_tbl_ptrs[n] != NULL) {
      huff_ptr = cinfo.dc_huff_tbl_ptrs[n];
      mxSetN(mxhuff_tables, n+1);

      mxtemp = mxCreateDoubleMatrix(1, 16, mxREAL);
      mxSetField(mxhuff_tables,n,"counts",mxtemp);
      mp = mxGetPr(mxtemp);
      for (i = 1; i <= 16; i++) *mp++ = huff_ptr->bits[i];

      mxtemp = mxCreateDoubleMatrix(1, 256, mxREAL);
      mxSetField(mxhuff_tables,n,"symbols",mxtemp);
      mp = mxGetPr(mxtemp);
      for (i = 0; i < 256; i++) *mp++ = huff_ptr->huffval[i];
    }
  }

  /* creation and population of the DCT coefficient arrays */
  coef_arrays = jpeg_read_coefficients(&cinfo);
  mxcoef_arrays = mxCreateCellMatrix(1,cinfo.num_components);
  mxSetField(mxjpeg_obj,0,"coef_arrays",mxcoef_arrays);
  for (ci = 0; ci < cinfo.num_components; ci++) {
    compptr = cinfo.comp_info + ci;
    c_height = compptr->height_in_blocks * DCTSIZE;
    c_width = compptr->width_in_blocks * DCTSIZE;

    /* create Matlab dct block array for this component */
    mxtemp = mxCreateDoubleMatrix(c_height, c_width, mxREAL);
    mxSetCell(mxcoef_arrays,ci,mxtemp);
    mp = mxGetPr(mxtemp);
    mptop = mp;

    /* copy coefficients from virtual block arrays */
    for (blk_y = 0; blk_y < compptr->height_in_blocks; blk_y++) {
      buffer = (cinfo.mem->access_virt_barray)
    	((j_common_ptr) &cinfo, coef_arrays[ci], blk_y, 1, FALSE);
      for (blk_x = 0; blk_x < compptr->width_in_blocks; blk_x++) {
        bufptr = buffer[0][blk_x];
        for (i = 0; i < DCTSIZE; i++)        /* for each row in block */
          for (j = 0; j < DCTSIZE; j++)      /* for each column in block */
            mp[j*c_height+i] = (double) bufptr[i*DCTSIZE+j];
        mp+=DCTSIZE*c_height;
      }
      mp=(mptop+=DCTSIZE);
    }
  }

  /* done with cinfo */
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  /* close input file */
  fclose(infile);

  /* set output */
  if (nlhs == 1) plhs[0] = mxjpeg_obj;

}
