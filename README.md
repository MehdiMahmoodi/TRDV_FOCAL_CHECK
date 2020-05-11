# Image QC Trimble Corporation_BeenaVision
--------------------------------------------
This repo contains the Image quality control for using in production line of image processing camera boxex.
This scanner box contain 3 different cameras: one Mikrotron, and 2 GT_1930.
This repo is containg the QC software for all camera which included in scanner box
The most chalenging part in Mikrotron focal check.

##Mikrotron Software

** Focal check software for TRDV Mikrotron camera, is working based on the logic:**

	- All images needs to be contained the 3-beam laser
	
	- The software will create the mask for each laser beam
	
	- All 3 beams will copy to the new image
	
	- the thickness of each beam will be calculated
	
	- The trend of laser beam thickness change will be visulize 
	
	- The trend of the laser will be saved for each beam in pdf file as the out put of the software
	
	- You can find the code [Here](https://github.com/MehdiMahmoodi/TRDV_Focal_check/blob/master/TRDV-Focal_check%20_my_pattern1.py)
