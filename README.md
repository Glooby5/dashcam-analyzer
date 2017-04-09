# Analýza záznamu palubní kamery vozidla
 - detekce zákazových dopravních značek
 - zmapování pokrytí dopravního značení na základě videozáznamu s dostupnými GPS daty
 
![Mapa dopravního značení](/excel/images/mapa.png)

## Excel@FIT
 - v rámci přípravy bakalářské práce vznikl krátký článek do soutěže [Excel@FIT](http://excel.fit.vutbr.cz) 
 - [PDF k nahlednutí](excel/excel-dashcam-analyzer-xkader13.pdf) 

 
## Trénování
 - `opencv_createsamples -info info.dat -nuum 440 -w 50 -h 50 -vec samples.vec -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5`
 - `opencv_traincascaded -data training4 -vec samples.vec -bg bg.txt -numStages 10 -numPos 420 -numNeg 800 -w 50 -h 50 -featureType LBP -maxFalseAlarmRate 0.3)`
 
## Detekce
 - `python detect.py -f="cascade.xml" -v="video.mp4"`

## Vytvoření videa

 - `video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'),15,(1280,720))`