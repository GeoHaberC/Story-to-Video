import os
import re
import srt
import sys
import math
import openai
import openai.error
import logging
import datetime
import requests
import traceback
import subprocess
import numpy as np
from srt import Subtitle

from typing import List, Optional, Tuple, Union
from My_Utils import perf_monitor, hm_sz, hm_time
from gtts import gTTS
from pydub import AudioSegment
from PIL import Image
from skimage import img_as_ubyte
from pydub import AudioSegment

# Constants
story_filename = "story.txt"
num_ChatGPT_images = 1
size_ChatGPT_image = "512x512"
frame_rate = 30

output_dir = 'output'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

@perf_monitor
def load_story_from_file(filename: str) -> Optional[List[str]]:
	''' load text line from stort file '''
	msj = sys._getframe().f_code.co_name

	if not os.path.isfile(filename):
		print(f"Story file '{filename}' not found. Please provide a valid story file.")
		filename = input("Enter the path to the story file: ")

	if not os.path.isfile(filename):
		print("Invalid file path. Please make sure the file exists and try again.")
		return None

	with open(filename, "r") as f:
		story = f.read().splitlines()
	msj += f" {story}"
	logging.info( msj )
	print( msj )

	return story

@perf_monitor
def generate_and_download_image(prompt: str, image_file: str, output_dir: str) -> Optional[str]:
	'''Generate an image using OpenAI's DALL-E API and download it to a local file.'''
	msj = sys._getframe().f_code.co_name
	logging.info(f"{msj} Text: {prompt}")

	try:
		response = openai.Image.create(
			prompt=prompt,
			n=num_ChatGPT_images,
			size=size_ChatGPT_image,
			response_format="url",
		)
		image_url = response["data"][0]["url"]

		# Download the image
		logging.info(f"Download from: {image_url} to File {image_file}")

		response = requests.get(image_url)
		if not response :
			raise (" No image created")

		with open(image_file, "wb") as f:
			f.write(response.content)
		logging.info(f"Downloaded image {image_file}")

		return image_file

	except openai.error.InvalidRequestError as e:
		print(f"Request for prompt '{prompt}' was rejected: {e}")
		logging.error(f"Request for prompt '{prompt}' was rejected by the safety system.")
		return None
	except Exception as e:
		print(f"An error occurred while generating an image for '{prompt}': {e}")
		logging.error(f"An error occurred while generating an image for '{prompt}': {e}\n{traceback.format_exc()}")
		return None

@perf_monitor
def generate_audio(text: str, cnt : int):
	''' Audio from text '''
	msj = sys._getframe().f_code.co_name
	msj += f" {cnt} Audio from {text}"
	logging.info(msj)
	print (msj)
	tts = gTTS(text=text, lang='en')
	audio_file = f"{output_dir}/{cnt:04d}_audio.mp3"
	tts.save(audio_file)
	return audio_file

@perf_monitor
def create_audio_file(audio_files : str, output_dir: str) -> str :
	''' combine audiofiles into one '''
	msj = sys._getframe().f_code.co_name
	# Combine audio files
	merged_audio_file = f"{output_dir}"
	merged_audio = AudioSegment.empty()
	for audio_file in audio_files:
		audio_segment = AudioSegment.from_file(audio_file)
		merged_audio = merged_audio + audio_segment
#        print(audio_file, audio_segment.frame_rate, audio_segment.channels, audio_segment.sample_width)
	merged_audio.export(merged_audio_file, format="mp3")
	msj += f" Merged audio file: {merged_audio_file}"
	logging.info( msj )
	print( msj )
	return merged_audio_file

@perf_monitor
def get_audio_length(audio_file: str, extra: int=0 ) -> int:
	''' Get audio lenght + extra'''
	msj = sys._getframe().f_code.co_name
	audio = AudioSegment.from_mp3(audio_file)
	au_len = math.ceil(audio.duration_seconds + extra)
	logging.info(f"{msj} Calc Audio length + {extra} = {au_len} sec")
	print (f"{msj} Calc Audio length + {extra} = {au_len} sec")
	return au_len

@perf_monitor
def morph_images(image1: str, image2: str, steps: int = 10, output_dir: str = ".", start_index: int = 0) -> Optional[List[str]]:
    ''' morph images in number of steps'''
    msj = sys._getframe().f_code.co_name
    msj += f" from: {image1} to: {image2} in {steps} Steps"
    logging.info(msj)
    print(msj)
    try:
        with Image.open(image1) as img1, Image.open(image2) as img2:
            img1_array = np.array(img1, dtype=np.float32) / 255.0
            img2_array = np.array(img2, dtype=np.float32) / 255.0

        morphed_images = []
        for cnt in range(steps + 1):
            alpha = cnt / steps
            morphed_image = (1 - alpha) * img1_array + alpha * img2_array

            # Save the morphed image to the disk
            filename = f"{output_dir}/morphed_image_{start_index + cnt:04d}.png"
            img = Image.fromarray(img_as_ubyte(morphed_image))
            img.save(filename)
            morphed_images.append(filename)

        return morphed_images
    except Exception as e:
        logging.error(f"An error occurred while morphing images: {e}\n{traceback.format_exc()}")
        return []

@perf_monitor
def create_video_file ( morphed_image_sequences: list , output_file: str ) :
	''' create a video file from static images '''
	msj = sys._getframe().f_code.co_name
	msj += f" Morphed images to video: {output_file}"
	logging.info (msj)
	print ( msj )

	msj = sys._getframe().f_code.co_name
#	cmd = f"ffmpeg -framerate 1 -i {output_dir}/morphed_image_%04d.png -c:v libx264 -r {frame_rate} -pix_fmt yuv420p -y {output_file}"
#	cmd = f"ffmpeg -framerate {frame_rate} -i {output_dir}/morphed_image_%04d.png -c:v libx265 -pix_fmt yuv420p -r {frame_rate} -y {output_file}"
	cmd = f"ffmpeg -report -i {output_dir}/morphed_image_%04d.png -safe 0 -c:v libx265 -preset slow -crf 25 -framerate {frame_rate} -y {output_file}"
	print(f"Executing command: {cmd}")
	msj += cmd
	logging.info(msj)
	print(msj)
	subprocess.run(cmd, shell=True, check=True)

	return output_file

@perf_monitor
def create_srt_file(subtitle_entries, output_file):
	''' Create subtitle bile from subtitiles'''
	msj = sys._getframe().f_code.co_name
	msj += f" subtitle_entries -> File: {output_file}"
	logging.info (msj)
	print ( msj )

	subtitles = []
	start_time = datetime.timedelta()
	for index, entry in enumerate(subtitle_entries):
		duration = datetime.timedelta(seconds=entry["duration"])
		end_time = start_time + duration
		content = make_legal_content(entry['text'])
		subtitle = Subtitle(index, start_time, end_time, content)
		subtitles.append(subtitle)
#		print (f"{index} In= {entry} txt ={entry['text']} Cn: {content} Sub: {subtitle} ")
		start_time = end_time

	with open(output_file, "w") as f:
		f.write(srt.compose(subtitles))

	return output_file

@perf_monitor
def add_subtitles_ffmpeg(video_file, srt_file, output_file):
	''' add subtile to video file '''
	msj = sys._getframe().f_code.co_name
	logging.info(f" ffmpeg + subtitle")
	cmd = f"ffmpeg -i {video_file} -i {srt_file} -vf subtitles={srt_file} -c:v libx264 -c:a copy -y {output_file}"
	msj += cmd
	print (msj)
	logging.info(f"{msj} File: {output_file}")
	subprocess.call(cmd, shell=True)

@perf_monitor
def make_legal_content(content):
	"""
	Replace any illegal characters with spaces and remove extra spaces.
	"""
	msj = sys._getframe().f_code.co_name

	return re.sub(r'[^\w\s]', ' ', content).strip()

@perf_monitor
def main(story_file):
	images = []
	audio_files = []
	subtitle_entries = []
	start_time = 0

	for cnt, line in enumerate(story_file):

		image_file = f"{output_dir}/image_{cnt:04d}.png"
		print(f"\n{cnt} Line {line} -> {image_file}")

		## XXX:  Generate and download image # XXX:
#		image_file = generate_and_download_image(line, image_file , output_dir)
		images.append(image_file)

		# Generate audio measure duration
		audio_file = generate_audio(line, cnt)
		duration = get_audio_length(audio_file)
		audio_files.append(audio_file)

		# subtitle entries
		end_time = start_time + duration
		subtitle_entries.append({
			"start": f"{start_time:.3f}",
			"end": f"{end_time:.3f}",
			"text": line,
			"duration": duration })  # Add duration to the dictionary
		start_time = end_time

		# Morph images
		morphed_image_sequences = []
		morph_steps = duration *frame_rate
		if cnt :
			morphed_images = morph_images(images[cnt - 1], images[cnt], morph_steps, output_dir, start_index=(cnt - 1) * morph_steps)
			morphed_image_sequences.extend(morphed_images)

	# Create video from images
	output_file   = f"{output_dir}/_only_video.mp4"
	morf_img_list = f"{output_dir}/image_list.txt"


	with open(morf_img_list, "w") as f:
		for cnt in range(len(morphed_image_sequences)):
			abs_image_path = os.path.abspath(f"{output_dir}/morphed_image_{cnt:04d}.png")
			print (abs_image_path)
			f.write(f"file '{abs_image_path}'\n")
	print ( morf_img_list )
#	cmd = f"ffmpeg -report -i {output_dir}/morphed_image_%04d.png -start_number 0 -c:v libx265 -preset slow -crf 25 -framerate {frame_rate} -y {output_file}"
	cmd = f"ffmpeg -report -framerate {frame_rate} -i {output_dir}/morphed_image_%04d.png -c:v libx265 -preset slow -crf 25 -vf fps={frame_rate} -y {output_file}"

#	cmd = f"ffmpeg -report -f concat -safe 0 -i {morf_img_list} -c:v libx265 -preset slow -crf 25 -framerate {frame_rate} -y {output_file}"
	subprocess.run(cmd, shell=True, check=True)

	# Merge audio files
	print(f"\n{len(audio_files)} Audio files")
	merged_audio = f"{output_dir}/merged_audio.mp3"
	aud_all_merged = create_audio_file(audio_files, merged_audio)

	# Create subtitle file
	print(f"\n{len(subtitle_entries)} Srt files")
	srt_file = f"{output_dir}/subtitles.srt"
	srt_all_merged = create_srt_file(subtitle_entries, srt_file)

	# Add audio and subtitles to video
	print(f"\nCobine images Audio and Subtitle files")
	video_audio_srt = f"{output_dir}/video_audio_srt.mp4"
	cmd = f"ffmpeg -report -framerate {frame_rate} -i {output_dir}/morphed_image_%04d.png -i {aud_all_merged} -i {srt_all_merged} \
		-map 0:v -c:v libx265 -preset slow -crf 25 \
		-map 1:a -c:a copy -metadata:s:a:0 language=eng \
		-map 2:s -c:s mov_text -metadata:s:s:0 language=eng -disposition:s:s:0 +default+forced \
		-y {video_audio_srt}"
	print(f"Executing command: {cmd}")
	print(f"Input file names: {output_dir}/morphed_image_*.png, {aud_all_merged}, {srt_all_merged}")
	subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
	# XXX: Setup the loggigng file
	name, _ = os.path.splitext(sys.argv[0])
	name += '_.log'
	# Set up logging to file
	logging.basicConfig(
			level=logging.INFO,
#			level=logging.DEBUG,
			filename=(name), filemode='w',
			format='%(asctime)s %(levelname)s %(message)s',
			datefmt='%d-%H:%M:%S')

	story_file = load_story_from_file(story_filename)
	main(story_file)
