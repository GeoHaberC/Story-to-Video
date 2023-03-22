# Story-to-Video
Create a Movie animation plus audio plus Subtitle from text
Text to Video Generator

This program generates a video from text using OpenAI's ChatGPT to create a sequence of images and gTTS (Google Text-to-Speech) to produce audio. The images and audio are then combined to create the final video.

1) Installation

- Ensure you have Python 3.6 or higher installed on your system.
- Install the required libraries by running the following command in your terminal or command prompt:

pip install openai requests gtts pydub scikit-image pillow numpy psutil

- Install FFmpeg, which is used for video processing. Visit the official FFmpeg website for installation instructions for your specific operating system.

https://ffmpeg.org/download.html

2) Usage

Run the Python script and follow the prompts to input the text file by default story.txt containing the story you would like to convert to video.
The program will generate images and audio based on your input and combine them into a video.
The Image files generated by oppen.ai are downloaded to thworking directory while the intermediate files and the resulting video to ../output 

3) Enhancements and Customizations
You can improve the program or customize it to suit your needs by:

- Add subtitles with the text synchronized to audio 
- Using alternative image generation models, such as StyleGAN, BigGAN, or DALLE-2, to create more diverse or higher-quality images.
- Incorporating different text-to-speech (TTS) libraries, such as Tacotron 2 or Mozilla's TTS, to generate audio with varied voices, accents, or intonations.
- Adjusting the program parameters (e.g., frame rate, video duration, audio quality) to optimize the output video.

4) Contributing
Feel free to contribute to this project by suggesting improvements, reporting bugs, or implementing new features. You can submit your changes via pull requests or open issues on the project repository.

Enjoy your text-to-video conversion!
