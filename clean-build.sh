#!/bin/bash

# directory where current shell script resides
PROJECTDIR=$(dirname "$BASH_SOURCE")

cd "$PROJECTDIR"

mode=$1 # debug/release mode
shift   # shift command-line arguments
        # the rest are cmake command-line arguments

mkdir -p build && cd build

# if debug directory does not exist, create it
mkdir -p debug
# if release directory does not exist, create it
mkdir -p release

args=("$@")

# if debug mode
if [ "$mode" = "debug" ]; then
    cd debug
# if release mode
elif [ "$mode" = "release" ]; then
    cd release
elif [ "$mode" = "test" ]; then
    cd debug
    args+=("-DTITAN_BUILD_TESTS=ON")
else
    echo "usage: $0 <debug/release/test> [cmake options]" 1>&2
    exit 1
fi

rm -rf *
cmake ../../ "${args[@]}"
cmake --build . -- -j12

read -p "Do you want to install Titan globally? [y/N] " -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    sudo make install
    sudo chmod -R 755 /usr/local
fi
