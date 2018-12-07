# To easily setup the development environment
init: 
	mkdir -p external
	mkdir -p weights
	mkdir -p datasets
	mkdir -p logs
deps-ubuntu:
	sudo apt-get install build-essential cmake pkg-config wget curl git tar
	sudo apt-get install gcc-5 g++-5
	sudo apt-get install python3 python3-dev python3-numpy python3-pip
	sudo apt-get install libatlas-base-dev liblapack-dev libblas-dev
	sudo apt-get install ffmpeg libpng-dev libjpeg-dev libtiff-dev libwebp-dev 
	sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
	sudo apt-get install libxvidcore-dev libx264-dev
	sudo apt-get install libgtk-3-dev

deps-python:
	pip3 install -r py3-requirements.txt

dlib:
	wget -O external/dlib-19.15.tar.bz2 -nd http://dlib.net/files/dlib-19.15.tar.bz2
	tar -xf external/dlib-19.15.tar.bz2
	mv dlib-19.15 external/dlib
	cd external/dlib && python3 setup.py install

opencv:
	git clone --branch 3.4.1 --depth 1 -o external/opencv https://github.com/opencv/opencv.git external/opencv
	git clone --branch 3.4.1 --depth 1 https://github.com/opencv/opencv_contrib.git external/opencv_contrib
	mkdir -p external/opencv/build
	cmake -D CMAKE_BUILD_TYPE=Release -D OPENCV_EXTRA_MODULES_PATH=external/opencv_contrib/modules -DCMAKE_INSTALL_PREFIX=/usr/local external/opencv
	cd external/opencv/build make -j4
	cd external/opencv/build && sudo make install
