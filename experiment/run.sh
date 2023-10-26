#! /bin/sh

for folder in groupon nhanes horse synthetic_7_0 synthetic_7_1 synthetic_7_2 synthetic_7_3 synthetic_7_4 synthetic_7_5 synthetic_7_6 synthetic_7_7 synthetic_7_8 synthetic_7_9 synthetic_7_10;
do cd $folder;
   # cp ../compare_propensity_matching.py .;
   python compare_propensity_matching.py;
   cd ..;
done