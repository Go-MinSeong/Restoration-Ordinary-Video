from info import *
import argparse

def main():
    if args.task == "video_to_frame":
        resize_video(input_path = args.video_path, output_path = args.video_path+"_M", target_resolution = (360, 360))
        change_fps(input_path = args.video_path+"_M" , output_path = args.video_path+"_M", target_fps = 30)
        extract_frames(input_path = args.video_path+"_M", output_folder  = args.frame_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task', type=str, default='video_to_frame', help='choose fuction.')
    parser.add_argument(
        '--video_path', type=str)
    parser.add_argument(
        '--frame_path', type=str)
    args = parser.parse_args()
    main()

