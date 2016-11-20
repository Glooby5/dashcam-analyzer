# Analýza záznamu palubní kamery vozidla
 - detekce značek omezujících rychlost
 - upozornění na místa s pravděpodobným překročením rychlosti
 
## Trénování
 - `opencv_createsamples -info info.dat -nuum 440 -w 50 -h 50 -vec samples.vec -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5`
 - `opencv_traincascaded -data training4 -vec samples.vec -bg bg.txt -numStages 10 -numPos 420 -numNeg 800 -w 50 -h 50 -featureType LBP -maxFalseAlarmRate 0.3)`
 
## Detekce
 - `python detect.py -f="cascade.xml" -v="video.mp4"`