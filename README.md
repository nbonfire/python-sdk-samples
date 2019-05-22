This script is the python-sdk-sample for frame detector.

Steps to install the app:

1. Make sure you have affvisionpy

2. cd into the directory containing the setup.py script and run: sudo python3 setup.py install.

   Alternative to step 2 : cd into the directory having the requirements.txt file and run "pip3 install -r requirements.txt"


Steps to run the script:

1. usage: 
    
        python-sample.py [-h] -d DATA [-i INPUT] [-n NUM_FACES]
        
        required arguments:
        
            -d DATA, --data DATA  path to directory containing the models
            

        optional arguments:
    
          -h, --help            show this help message and exit
      
          -i INPUT, --input INPUT
                        path to input video file
                        
          -n NUM_FACES, --num_faces NUM_FACES
                        number of faces to identify in the frame
                        
        Note: if only data argument is supplied, the script defaults the run to a webcam and 1 face detection. If any other configuration is required, it can be done using optional arguments.
        


        


2. We can use the same script to enable camera as well as input video.

3. By default the script runs the webcam and the num of faces it detects is 1.

4. 

    i. Command to run the script with webcam: 

            python3 python-sample.py -d <path/to/data/directory> -n <num_of_faces_to_detect>
        
    ii. Command to run the script with a video file:
    
            python3 python-sample.py -d <path/to/data/directory> -n <num_of_faces_to_detect> -i </path/to/video/file>
        
Note:

1. The file writes all metrics(expressions, emotions, head angles) to a csv file known as "output.csv".

2. It also writes the individual frames numbered by the framecount in a directory called as "opvideo".

3. The video file is written to a file called as output.avi by the script.
    
   