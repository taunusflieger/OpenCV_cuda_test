#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaimgproc.hpp"
//#include "opencv2/core/cuda.hpp"
//#include "opencv2/imgproc.hpp"

using namespace cv;

int main(int argc, char** argv ) {
  if ( argc != 2 ) {
    printf("usage: resizegpu <Image_Path>\n");
    return -1;
  }
  Mat inImage;
  Mat outImage;
  inImage = imread( argv[1], 1 );
  if ( !inImage.data ) {
    printf("No image data \n");
    return -1;
  }

  cuda::GpuMat gpuInImage;
  cuda::GpuMat gpuOutImage;

  for(int i=0; i < 100; i++){
    //resize
    gpuInImage.upload(inImage);

    cv::cuda::resize(gpuInImage, gpuOutImage, Size(4096, 4096));

    gpuOutImage.download(outImage);

    imwrite("output_gpu.jpg", outImage);
  }

  return 0;
}

