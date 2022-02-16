# Thesis 2021: VIDEO SUMMARIZING AT THE HAPPIEST MOMENT VIA FACE BEAUTIFICATION
# 1. Download conda
See [this instruction](https://github.com/user/repo/blob/branch/other_file.md) to download conda.
# 2. Set up environment
* Download source code:
```
git clone https://github.com/tuandq-cs/thesis-2021.git
```
* Install dependences and create a new conda environment:
```
conda env create -f environment.yml
```
# 3. Get pretrained models
* Set current working directory to root:
```
cd thesis-2021/
```
* Install [pretrained model](https://drive.google.com/file/d/1LhLImu75EcAoSeD4Q3RjzTtX__2luWkb/view?usp=sharing) 
* Extract file:
```
unzip models.zip
```
# 4. Run
* Activate the environment:
```
conda activate $YOUR_ENV_NAME
```
* Finally, put your video path into this script: 
```
python main.py --video_path=$YOUR_VIDEO_PATH
```
* Result will be store in inference folder ```inferences/ ```
* Folders in inference folder are formated by: ```year_month_day_hour_min```
* The lastest folder is our result.

# Alternatives
**Or make it easier by looking at this** [**Google Colab file**](https://colab.research.google.com/drive/1yv7bv4ee52heF0Lk388RlI5fCu9_07N1?usp=sharing) 
