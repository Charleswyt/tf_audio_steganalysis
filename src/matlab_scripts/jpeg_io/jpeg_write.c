/*
 * jpeg_write.c
 *
 * JPEG_WRITE(JPEGOBJ,FILENAME)
 *
 * Reads JPEGOBJ, a Matlab struct containing the JPEG header, 
 * quantization tables and the DCT coefficients (as returned by JPEG_READ),
 * and writes the information into a JPEG file with the name FILENAME.
 *
 * This software is based in part on the work of the Independent JPEG Group.
 * In order to compile, you must first build IJG's JPEG Tools code library, 
 * available at ftp://ftp.uu.net/graphics/jpeg/jpegsrc.v6b.tar.gz.
 *
 * Phil Sallee, Surya De 6/2003
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
#include <jpegint.h>
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


/* mex function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  struct jpeg_compress_struct cinfo;
  struct my_error_mgr jerr;
  const mxArray *mxjpeg_obj;
  mxArray *mxcoef_arrays,*mxhuff_tables,*mxcomp_info,*mxtemp,
          *mxquant_tables,*mxcomments;
  char *filename,*comment;
  int strlen,c_height,c_width,ci,i,j,n,t;
  FILE *outfile;
  jvirt_barray_ptr *coef_arrays = NULL;
  jpeg_component_info *compptr;
  JDIMENSION blk_x,blk_y;
  JBLOCKARRAY buffer;
  JCOEFPTR bufptr;  
  double *mp, *mptop;

  /* check the input values */
  if (nrhs != 2) mexErrMsgTxt("Two input arguments required.");

  /* check the output values */
  if (nlhs != 0) mexErrMsgTxt("Too many output arguments.");
  if (mxIsChar(prhs[1]) != 1) mexErrMsgTxt("Filename must be a string.");

  /* get filename */
  strlen = mxGetM(prhs[1])*mxGetN(prhs[1]) + 1;
  filename = mxCalloc(strlen, sizeof(char));
  mxGetString(prhs[1],filename,strlen);

  /* open the output file*/
  if ((outfile = fopen(filename, "wb")) == NULL)
    mexErrMsgTxt("Can't open file.");

  /* set up the normal JPEG error routines, then override error_exit. */
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = my_error_exit;
  jerr.pub.output_message = my_output_message;

  /* establish the setjmp return context for my_error_exit to use. */
  if (setjmp(jerr.setjmp_buffer))
  {
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
    mexErrMsgTxt("Error writing to file.");
  }

  /* set the input */
  mxjpeg_obj = prhs[0];

  /* initialize JPEG decompression object */
  jpeg_create_compress(&cinfo);

  /* write the output file */
  jpeg_stdio_dest(&cinfo, outfile);

  /* Set the compression object with our parameters */
  cinfo.image_width = 
    (unsigned int) mxGetScalar(mxGetField(mxjpeg_obj,0,"image_width"));
  cinfo.image_height = 
    (unsigned int) mxGetScalar(mxGetField(mxjpeg_obj,0,"image_height"));
  cinfo.input_components = 
    (int) mxGetScalar(mxGetField(mxjpeg_obj,0,"image_components"));
  cinfo.in_color_space = 
    (int) mxGetScalar(mxGetField(mxjpeg_obj,0,"image_color_space"));

  /* set the compression object with default parameters */
  jpeg_set_defaults(&cinfo);

  cinfo.optimize_coding =
    (unsigned char) mxGetScalar(mxGetField(mxjpeg_obj,0,"optimize_coding"));
  cinfo.num_components =
    (int) mxGetScalar(mxGetField(mxjpeg_obj,0,"jpeg_components"));
  cinfo.jpeg_color_space =
    (int) mxGetScalar(mxGetField(mxjpeg_obj,0,"jpeg_color_space"));

  /* basic support for writing progressive mode JPEG */
  if (mxGetField(mxjpeg_obj,0,"progressive_mode")) {
    if ((int) mxGetScalar(mxGetField(mxjpeg_obj,0,"progressive_mode")))
      jpeg_simple_progression(&cinfo);
  }

  /* obtain the component array from the jpeg object  */
  mxcomp_info = mxGetField(mxjpeg_obj,0,"comp_info");

  /* copy component information into cinfo from jpeg_obj*/
  for (ci = 0; ci < cinfo.num_components; ci++)
  {
    cinfo.comp_info[ci].component_id =
      (int) mxGetScalar(mxGetField(mxcomp_info,ci,"component_id"));
    cinfo.comp_info[ci].h_samp_factor =
      (int) mxGetScalar(mxGetField(mxcomp_info,ci,"h_samp_factor")); 
    cinfo.comp_info[ci].v_samp_factor =
      (int) mxGetScalar(mxGetField(mxcomp_info,ci,"v_samp_factor"));
    cinfo.comp_info[ci].quant_tbl_no =
      (int) mxGetScalar(mxGetField(mxcomp_info,ci,"quant_tbl_no"))-1;
    cinfo.comp_info[ci].ac_tbl_no =
      (int) mxGetScalar(mxGetField(mxcomp_info,ci,"ac_tbl_no"))-1;
    cinfo.comp_info[ci].dc_tbl_no =
      (int) mxGetScalar(mxGetField(mxcomp_info,ci,"dc_tbl_no"))-1;
  }


  /* request virtual block arrays */
  mxcoef_arrays = mxGetField(mxjpeg_obj, 0, "coef_arrays");  
  coef_arrays = (jvirt_barray_ptr *)
    (cinfo.mem->alloc_small) ((j_common_ptr) &cinfo, JPOOL_IMAGE,
     sizeof(jvirt_barray_ptr) * cinfo.num_components);
  for (ci = 0; ci < cinfo.num_components; ci++)
  {
    compptr = cinfo.comp_info + ci;
  
    c_height = mxGetM(mxGetCell(mxcoef_arrays,ci));
    c_width = mxGetN(mxGetCell(mxcoef_arrays,ci));
    compptr->height_in_blocks = c_height / DCTSIZE;
    compptr->width_in_blocks = c_width / DCTSIZE;
  
    coef_arrays[ci] = (cinfo.mem->request_virt_barray)
      ((j_common_ptr) &cinfo, JPOOL_IMAGE, TRUE,
       (JDIMENSION) jround_up((long) compptr->width_in_blocks,
                              (long) compptr->h_samp_factor),
       (JDIMENSION) jround_up((long) compptr->height_in_blocks,
                              (long) compptr->v_samp_factor),
       (JDIMENSION) compptr->v_samp_factor);
  }

  
  /* realize virtual block arrays */
  jpeg_write_coefficients(&cinfo,coef_arrays);

  /* populate the array with the DCT coefficients */
  for (ci = 0; ci < cinfo.num_components; ci++)
  {
    compptr = cinfo.comp_info + ci;

    /* Get a pointer to the mx coefficient array */
    mxtemp = mxGetCell(mxcoef_arrays,ci);
    mp = mxGetPr(mxtemp);
    mptop = mp;
    
    c_height = mxGetM(mxtemp);
    c_width = mxGetN(mxtemp);

    /* Copy coefficients to virtual block arrays */
    for (blk_y = 0; blk_y < compptr->height_in_blocks; blk_y++)
    {
      buffer = (cinfo.mem->access_virt_barray)
  	((j_common_ptr) &cinfo, coef_arrays[ci], blk_y, 1, TRUE);

      for (blk_x = 0; blk_x < compptr->width_in_blocks; blk_x++)
      {
        bufptr = buffer[0][blk_x];
        for (i = 0; i < DCTSIZE; i++)        /* for each row in block */
          for (j = 0; j < DCTSIZE; j++)      /* for each column in block */
            bufptr[i*DCTSIZE+j] = (JCOEF) mp[j*c_height+i];
        mp+=DCTSIZE*c_height;
      }
      mp=(mptop+=DCTSIZE);
    }
  }

  /* get the quantization tables */
  mxquant_tables = mxGetField(mxjpeg_obj,0,"quant_tables");
  for (n = 0; n < mxGetN(mxquant_tables); n++)
  {
    if (cinfo.quant_tbl_ptrs[n] == NULL)
      cinfo.quant_tbl_ptrs[n] = jpeg_alloc_quant_table((j_common_ptr) &cinfo);

    /* Fill the table */
    mxtemp = mxGetCell(mxquant_tables,n);
    mp = mxGetPr(mxtemp);
    for (i = 0; i < DCTSIZE; i++) 
      for (j = 0; j < DCTSIZE; j++) {
        t = mp[j*DCTSIZE+i];

        if (t<1 || t>65535)
          mexErrMsgTxt("Quantization table entries not in range 1..65535");

        cinfo.quant_tbl_ptrs[n]->quantval[i*DCTSIZE+j] = 
          (UINT16) t;
      }
  }

  /* set remaining quantization table slots to null */
  for (; n < NUM_QUANT_TBLS; n++)
    cinfo.quant_tbl_ptrs[n] = NULL;

  /* Get the AC and DC huffman tables but check for optimized coding first*/
  if (cinfo.optimize_coding == FALSE)
  {
    if (mxGetField(mxjpeg_obj,0,"ac_huff_tables") != NULL)
    {
      mxhuff_tables = mxGetField(mxjpeg_obj,0,"ac_huff_tables");
      if( mxGetN(mxhuff_tables) > 0)
      {
        for (n = 0; n < mxGetN(mxhuff_tables); n++)
	{
          if (cinfo.ac_huff_tbl_ptrs[n] == NULL)
	    cinfo.ac_huff_tbl_ptrs[n] =
            jpeg_alloc_huff_table((j_common_ptr) &cinfo);
	  
          mxtemp = mxGetField(mxhuff_tables,n,"counts");
          mp = mxGetPr(mxtemp);
          for (i = 1; i <= 16; i++)
            cinfo.ac_huff_tbl_ptrs[n]->bits[i] = (UINT8) *mp++;
          mxtemp = mxGetField(mxhuff_tables,n,"symbols");
          mp = mxGetPr(mxtemp);
          for (i = 0; i < 256; i++)
            cinfo.ac_huff_tbl_ptrs[n]->huffval[i] = (UINT8) *mp++;
        }
        for (; n < NUM_HUFF_TBLS; n++) cinfo.ac_huff_tbl_ptrs[n] = NULL;
      }
    }
    
    if (mxGetField(mxjpeg_obj,0, "dc_huff_tables") != NULL)
    {
      mxhuff_tables = mxGetField(mxjpeg_obj,0,"dc_huff_tables");
      if( mxGetN(mxhuff_tables) > 0)
      {
        for (n = 0; n < mxGetN(mxhuff_tables); n++)
	{
          if (cinfo.dc_huff_tbl_ptrs[n] == NULL)
	    cinfo.dc_huff_tbl_ptrs[n] =
            jpeg_alloc_huff_table((j_common_ptr) &cinfo);

          mxtemp = mxGetField(mxhuff_tables,n,"counts");
          mp = mxGetPr(mxtemp);
          for (i = 1; i <= 16; i++)
            cinfo.dc_huff_tbl_ptrs[n]->bits[i] = (unsigned char) *mp++;
          mxtemp = mxGetField(mxhuff_tables,n,"symbols");
          mp = mxGetPr(mxtemp);
          for (i = 0; i < 256; i++)
            cinfo.dc_huff_tbl_ptrs[n]->huffval[i] = (unsigned char) *mp++;
        }
        for (; n < NUM_HUFF_TBLS; n++) cinfo.dc_huff_tbl_ptrs[n] = NULL;
      }
    }
  }
 
  /* copy markers */
  mxcomments =  mxGetField(mxjpeg_obj,0,"comments");
  n = mxGetN(mxcomments);
  for (i = 0; i < n; i++)
  {
     mxtemp = mxGetCell(mxcomments,i);
     strlen = mxGetN(mxtemp) + 1;
     comment = mxCalloc(strlen, sizeof(char));
     mxGetString(mxtemp,comment,strlen);
     jpeg_write_marker(&cinfo, JPEG_COM, comment, strlen-1);
     mxFree(comment);   
  }

  /* done with cinfo */
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  /* close the file */
  fclose(outfile);
}
