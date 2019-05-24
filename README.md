This script is the python-sdk-sample for frame detector.

Steps to install the app:

1. Make sure you have affvisionpy

2. cd into the directory containing the setup.py script and run: **sudo python3 setup.py install**.

   Alternative to step 2 : cd into the directory having the requirements.txt file and run **pip3 install -r requirements.txt**


Steps to run the script:

1. usage: 
    

        affvisionpy-sample.py [-h] -d DATA [-v VIDEO] [-n NUM_FACES] [-c [CAMERA]] [-o OUTPUT] [-f FILE]
        
        required arguments:
        
            -d DATA, --data DATA  path to directory containing the models
            

        optional arguments:
    
          -h, --help            show this help message and exit
      
          -i VIDEO, --input VIDEO
                        path to input video file
                        
          -n NUM_FACES, --num_faces NUM_FACES
                        number of faces to identify in the frame
                        
          -o OUTPUT, --output OUTPUT
                        name of the output video file
          
          -f FILE, --file FILE
                        name of the output csv file
                        
          -c [CAMERA], --camera [CAMERA]
                        enable this parameter take input from the webcam and provide a camera id for the webcam
                        
        Note: if only data argument is supplied, the script defaults the run to a webcam and 1 face detection. If any other configuration is required, it can be done using optional arguments.
        

2. We can use the same script to enable camera as well as input video.

3. By default the num of faces detected by the script is 1.

4. 

    i. Command to run the script with webcam: 

            python3 affvisionpy-sample.py -d <path/to/data/directory> -c <camera_id> -n <num_of_faces_to_detect>
            
            Note: If the camera id is not supplied, by default the camera_id 0 is taken
        
    ii. Command to run the script with a video file:
    
            python3 affvisionoy-sample.py -d <path/to/data/directory> -n <num_of_faces_to_detect> -v </path/to/video/file>
        
    
5. When a video file is provided as an input, a csv file is written with the metrics. By default the csv file will
    be named as the name of the video file.

6. When a webcam is used as an input, the script displays real-time metrics on the screen.

7. To get the output metrics in a file of your choice, give the name of the file with -f on the command line.

8. To get the output video, use the option -o along with the output file name.
