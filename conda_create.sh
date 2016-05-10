#make sure you've got the anaconda version of pip on your path
#use an old verison of firefox
#https://ftp.mozilla.org/pub/firefox/releases/30.0/mac/en-US/
#i've also added the "disable all images" plugin in firefox
packages="boto3 pandas qt"
conda create -n recipe-recommender $packages
source activate recipe-recommender
pip install selenium

