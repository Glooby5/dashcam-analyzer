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
 
## Sturktura projektu

 - src
    - analyzer - modelová část projektu, obsahuje implementaci všech použitých algoritmů
        - value_classification - zahrnuje nástroje pro přípravu klasifikátoru hodnot
            - generate_dataset.py - soubory v zadaných složkách zapíše do textového souboru
            - create_dataset.py - pokusí se roztřídit předaný dataset
            - train.py - připraví soubory s informaci o dostupných vzorcích pro klasifikaci
            - test.py - otestuje úspěšnost klasifikace
        - blobl.py - reprezentuje symbol v rámci obrazu značky rychlostního omezení
        - crop_sign.py - zapouzdřuje zúžení výřezu dopravní značky
        - frame.py - reprezentuje snímek v rámci videa
        - sign.py - obecná reprezentace detekované dopravní značky
        - smart_sign.py - reprezentace dopravní značky umožňující klasifikaci hodnoty rychlostního omezení
        - theshold_sign.py - zapouzdřuje proces vytváření binární reprezentace obrazu dopravní značky
        - type_sign.py - reprezentuje proces klasifikování typu dopravní značky
        - video_processor.py - zapozřudej rpoces načítání snímků z videa a vytváření interních reprezentací
    - cascade_test - zahrnuje prostředky pro otestování detektoru dopravních značek
        - anotate_frames.py - utilita pro vytvoření anotací značek v rámci snímku
        - save_frames.py - načte video a jeho snímky uloží jako samostatné soubory
        - test_frames.py - načítá snímky v předaném datasetu a podle předaného anotačního souboru testuje úspěšnost klasifikace
    - lib - soubory s obecnou funkcionalitou (analýza barevného modelu, vyhledávání značek pomocí analýzy barevného modelu...)
        - sign_detector.py - zahrnuje obecné operace nad obrazem
        - data_generator.py - pomocí základních metod analyzuje obraz
    - parser - prostředky pro zpracování záznamů o poloze GPS
        - srtparser - implementace
            - record.py - reprezentuje jeden záznam v rámci SRT souboru
            - srt_parser.py - zpracuje kompletní soubor s titulky
        - test - testy pro parsování titulků
    - sign_classification - zahrnuje nástroje pro přípravu klasifikátoru typu značky
        - generate_dataset.py - soubory v zadaných složkách zapíše do textového souboru
        - train.py - připraví soubory s informaci o dostupných vozrkách pro klasifikaci
        - test.py - otestuje úspěšnost klasifikace
    - detect.py - demostruje analýzu záznamu nad všemi snímky
    - export.py - provede analýzu záznamu a vytvoří výsledný exportovaný soubor s unikátními výskyty dopravních značek