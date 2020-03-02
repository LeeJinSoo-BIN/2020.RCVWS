#Normal
OMP_NUM_THREAD=1 python eval_coco.py

#Black_out
OMP_NUM_THREAD=1 python eval_coco.py --synth_fail blackout None
OMP_NUM_THREAD=1 python eval_coco.py --synth_fail None blackout

##Mix
OMP_NUM_THREAD=1 python eval_coco.py --synth_fail crack2.jpg bug2.png
OMP_NUM_THREAD=1 python eval_coco.py --synth_fail bug2.png crack2.jpg

OMP_NUM_THREAD=1 python eval_coco.py --synth_fail crack2.jpg dust2.png
OMP_NUM_THREAD=1 python eval_coco.py --synth_fail dust2.png crack2.jpg

OMP_NUM_THREAD=1 python eval_coco.py --synth_fail dust2.png bug2.png
OMP_NUM_THREAD=1 python eval_coco.py --synth_fail bug2.png dust2.png

OMP_NUM_THREAD=1 python eval_coco.py --synth_fail dust3.png dust2.png
OMP_NUM_THREAD=1 python eval_coco.py --synth_fail dust4.png dust3.png

OMP_NUM_THREAD=1 python eval_coco.py --synth_fail dust2.png dust3.png
OMP_NUM_THREAD=1 python eval_coco.py --synth_fail dust3.png dust4.png
### Crack1 ~ Crack3
for ii in {1..3}
do        
    failStr=$( printf 'crack%d.jpg' $ii )
    OMP_NUM_THREAD=1 python eval_coco.py --synth_fail $failStr None 
    OMP_NUM_THREAD=1 python eval_coco.py --synth_fail None $failStr
done

### Bug1 ~ Bug4
for ii in {1..4}
do        
    failStr=$( printf 'bug%d.png' $ii )
    OMP_NUM_THREAD=1 python eval_coco.py --synth_fail $failStr None
    OMP_NUM_THREAD=1 python eval_coco.py --synth_fail None $failStr
done    
        
###  Dust1 ~ Dust7
for ii in {1..7}
do        
    failStr=$( printf 'dust%d.png' $ii )
    OMP_NUM_THREAD=1 python eval_coco.py --synth_fail $failStr None
    OMP_NUM_THREAD=1 python eval_coco.py --synth_fail None $failStr
done   
