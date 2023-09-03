#!/bin/bash

# Download the text for training.
wget https://raw.githubusercontent.com/naosense/Yiya/master/book/%E4%B8%89%E5%9B%BD%E6%BC%94%E4%B9%89.txt -O sanguo.txt

# The text is encoded in GBK, let's convert it to UTF-8.
iconv -f GBK -t UTF-8 sanguo.txt > sanguo-utf8.txt

# Remove the irrelevant text (e.g. seperators between chapters)
# For Linux:
sed -i '/^=/,/^=/d' sanguo-utf8.txt
# For Mac:
# sed -i "" '/^=/,/^=/d' sanguo-utf8.txt
