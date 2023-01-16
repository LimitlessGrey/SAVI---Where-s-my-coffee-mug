# Object Finder 9000 

The Object Finder 9000 is the final project for the course " Sistemas Avançados de Visão Industrial", taught at the University of Aveiro in the first semester of the year 2022/2023.

Its main purpose is to locate objects from the Washington RGB-D Object Dataset into the Washington RGB-D Scenes Dataset.     
As such it is comprised of model trained do correctly classify the everyday objects present at the Washington RGB-D Object Dataset and then, recognize the objects in the Washington RGB-D Scenes Dataset.   
To do so, the coordinates of the objects were found in the Point Cloud that was associated with the correponding RGB image.  
Finally when the correct positions of the objects were found, the model previously trained is then put into practice to indetify this said objects.
## Authors

- [Francisco Stigliano](https://github.com/LimitlessGrey)
- [Igino Contin](https://github.com/Contin1999)
- [Rafael Oliveira](https://github.com/Rafafbo)



## Acknowledgements

 - [Miguel Riem de Oliveira](https://github.com/miguelriemoliveira)
 

## Features

- MultiClass Classification
- Object Detection
- Object Description
 


## About Us
We are Mechanical Engineering and Industrial Automation Engineering Students from the University of Aveiro. 

## Lessons Learned

In this project we learned :

>  + How to train a multiclass image classifier and apply it to a scene.
>  + How to get the centroids of the objects located in an image via its point cloud
>  + Apply transformation matrices to translate, transpose and/or rotate images or its coordinate axis.
## Appendix

The Datasets used can be downloaded from the [Washington RGB-D Dataset](http://rgbd-dataset.cs.washington.edu/dataset/)

## License

[MIT](https://choosealicense.com/licenses/mit/)  


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
