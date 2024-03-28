# Find-the-tag
## Description
The purpose of this project is to track staff members wearing a staff name tag.
<img width="222" alt="example" src="https://github.com/zampoan/Find-the-tag/assets/49183569/2c8f900f-c235-4eb2-a989-726c1ffd7ea8">

## Methodology
Originally the plan was to use a variation of centroid tracking from openCV tracker algorithms such as 'KCF' or 'MOSSE', however it proved to be ineffective. So a deep learning model 
was used instead.
The model that was used is YOLOv8m. To train the model, several reference images from sample.mp4 as well as several real-life pictures of myself. It was then annotated using labelImg. 
To see if the model was properly trained, a short video called sampleIRL.mp4 was tested.

## Output
The trained video of sample.mp4 can be found in the following directory: /runs/detect/predict2/sample.mp4
Whereas the trained video of sampleIRL.mp4 can be found: /runs/detect/predict/sample.mp4
When executing the program, please note that getting the frame at which the staff is present and the xy locations of staff members is a little slow due to the size of the array.

Example of the final output:
![output](https://github.com/zampoan/Find-the-tag/assets/49183569/d3694785-16c6-44a3-a317-e751a30d34dd)
