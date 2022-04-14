import time
start_time = time.time()
import argparse
import os
import sys
import os.path
import cv2
import extracting_candidate_frames
import clustering_with_hdbscan
import numpy as np
# from multiprocessing import Pool, Process, cpu_count
import logging
import glob


logging.basicConfig(filename='/content/drive/MyDrive/soco-keyframe-extraction/logs/key_frames.log',format='%(asctime)s  %(levelname)s:%(message)s',level=logging.DEBUG)
logging.info('---------------------------------------------------------------------------------------------------------')


def main(argv):

    files = glob.glob("/content/drive/MyDrive/Data_Visha/train/videos/*.mp4")

    for path in files:
      print(path)
      input_videos = path

      video_name=input_videos.rsplit( ".", 1 )[ 0 ].split('/')[-1]
      output_path = "/content/drive/MyDrive/Data_Visha/train/keyFrame_images/" + video_name
      output_folder_video_image = "/content/drive/MyDrive/Data_Visha/train/clusters/" + video_name
      output_folder_video_final_image = "final_images"
      
      logging.info('file execution started for input video {}'.format(input_videos))
      vd = extracting_candidate_frames.FrameExtractor()
      if not os.path.isdir(output_folder_video_image):
          os.makedirs(output_folder_video_image)
          os.makedirs(output_path)
      fps, imgs =vd.extract_candidate_frames(input_videos)
      # print("images:",len(imgs))
      # print("fps:",fps)
    
      """
      for counter, img in enumerate(imgs):
          _i, img = img[1], img[0]
          vd.save_frame_to_disk(
              img,
              file_path=os.path.join(input_videos.rsplit( ".", 1 )[ 0 ],output_folder_video_image),
              file_name=str(_i)+"_" + str(counter),
              file_ext=".jpeg",
          )
      """
      final_images = clustering_with_hdbscan.ImageSelector()
      imgs_final = final_images.select_best_frames(imgs,output_folder_video_image)
      #for counter, i, k in enumerate(imgs_final):
      for counter, i in enumerate(imgs_final):
        #print("counter:",counter)
        #print("i:",i)
        # print(str(np.zeros(8-len(str(i[2]))))+str(i[2]))
        vd.save_frame_to_disk(
            i[0],
            file_path= os.path.join(output_path),
            # file_path = os.path.join("/content/drive/MyDrive/Data_Visha/train/keyFrame_images", input_videos.rsplit( ".", 1 )[ 0 ]),
            #file_name=argv[1].split(".")[0]+"_"+str(duration)+"_"+str(len(imgs)+1)+"_" + str(i[1])+"_"+str(duration*int(i[1])/(len(imgs)+1)),
            file_name = str(i[2]).zfill(8),
            #file_name = , 
            file_ext=".jpg",
        )

if __name__ == "__main__":
    main(sys.argv)
