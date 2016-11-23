#!/bin/bash
export LC_CTYPE=C 
export LANG=C

function xclean {
	# Cleaning all the known XML tags
	cat $1 |sed 's/<\(url\|talkid\|keywords\)>.*<\/\1>//g' |sed 's/<[^>]*>//g'
}

function xcleanout {
	echo Creating ./clean/ directory
	mkdir -p clean	# Create clean/ directory
	echo Cleaning each file in ./
	for f in *{xml,tags}*
		do
		echo Cleaning file $f, clean file will be ./clean/$f.clean
		xclean $f >clean/$f.clean	# xCleaning each file
	done
	echo Done. Cleaned all '*xml* and *tags*' files in ./, clean files in ./clean/.
}

xcleanout	# Run the main function