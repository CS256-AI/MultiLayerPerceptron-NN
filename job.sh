#!/bin/sh
# create 5k training folder
mkdir -m777 train_5k
python sticky_snippet_generator.py 5000 0 0 nosticky.txt
python sticky_snippet_generator.py 2500 0 1 1sticky.txt
python sticky_snippet_generator.py 2500 0 2 2sticky.txt
python sticky_snippet_generator.py 2500 0 3 3sticky.txt
python sticky_snippet_generator.py 2500 0 4 4sticky.txt
python sticky_snippet_generator.py 2500 0 5 5sticky.txt
python sticky_snippet_generator.py 2500 0 6 6sticky.txt
python sticky_snippet_generator.py 2500 0 7 7sticky.txt
python sticky_snippet_generator.py 2500 0 8 8sticky.txt
python sticky_snippet_generator.py 5000 0 20 stickypal.txt

paste -d '\n' 1sticky.txt 2sticky.txt > sticky12.txt
paste -d '\n' 3sticky.txt 4sticky.txt > sticky34.txt
paste -d '\n' 5sticky.txt 6sticky.txt > sticky56.txt
paste -d '\n' 7sticky.txt 8sticky.txt > sticky78.txt
rm 1sticky.txt
rm 2sticky.txt
rm 3sticky.txt
rm 4sticky.txt
rm 5sticky.txt
rm 6sticky.txt
rm 7sticky.txt
rm 8sticky.txt

mv *.txt train_5k

# create 10k training folder
mkdir -m777 train_10k
python sticky_snippet_generator.py 10000 0 0 nosticky.txt
python sticky_snippet_generator.py 5000 0 1 1sticky.txt
python sticky_snippet_generator.py 5000 0 2 2sticky.txt
python sticky_snippet_generator.py 5000 0 3 3sticky.txt
python sticky_snippet_generator.py 5000 0 4 4sticky.txt
python sticky_snippet_generator.py 5000 0 5 5sticky.txt
python sticky_snippet_generator.py 5000 0 6 6sticky.txt
python sticky_snippet_generator.py 5000 0 7 7sticky.txt
python sticky_snippet_generator.py 5000 0 8 8sticky.txt
python sticky_snippet_generator.py 10000 0 20 stickypal.txt

paste -d '\n' 1sticky.txt 2sticky.txt > sticky12.txt
paste -d '\n' 3sticky.txt 4sticky.txt > sticky34.txt
paste -d '\n' 5sticky.txt 6sticky.txt > sticky56.txt
paste -d '\n' 7sticky.txt 8sticky.txt > sticky78.txt
rm 1sticky.txt
rm 2sticky.txt
rm 3sticky.txt
rm 4sticky.txt
rm 5sticky.txt
rm 6sticky.txt
rm 7sticky.txt
rm 8sticky.txt

mv *.txt train_10k

# create 20k training folder
mkdir -m777 train_20k
python sticky_snippet_generator.py 20000 0 0 nosticky.txt
python sticky_snippet_generator.py 10000 0 1 1sticky.txt
python sticky_snippet_generator.py 10000 0 2 2sticky.txt
python sticky_snippet_generator.py 10000 0 3 3sticky.txt
python sticky_snippet_generator.py 10000 0 4 4sticky.txt
python sticky_snippet_generator.py 10000 0 5 5sticky.txt
python sticky_snippet_generator.py 10000 0 6 6sticky.txt
python sticky_snippet_generator.py 10000 0 7 7sticky.txt
python sticky_snippet_generator.py 10000 0 8 8sticky.txt
python sticky_snippet_generator.py 20000 0 20 stickypal.txt

paste -d '\n' 1sticky.txt 2sticky.txt > sticky12.txt
paste -d '\n' 3sticky.txt 4sticky.txt > sticky34.txt
paste -d '\n' 5sticky.txt 6sticky.txt > sticky56.txt
paste -d '\n' 7sticky.txt 8sticky.txt > sticky78.txt
rm 1sticky.txt
rm 2sticky.txt
rm 3sticky.txt
rm 4sticky.txt
rm 5sticky.txt
rm 6sticky.txt
rm 7sticky.txt
rm 8sticky.txt

mv *.txt train_20k

# create 60k training folder
mkdir -m777 train_60k
python sticky_snippet_generator.py 60000 0 0 all.txt
mv *.txt train_60k

# create 5k 0.2 test folder
mkdir -m777 test_5k_2
python sticky_snippet_generator.py 5000 0.2 0 nosticky.txt
python sticky_snippet_generator.py 2500 0.2 1 1sticky.txt
python sticky_snippet_generator.py 2500 0.2 2 2sticky.txt
python sticky_snippet_generator.py 2500 0.2 3 3sticky.txt
python sticky_snippet_generator.py 2500 0.2 4 4sticky.txt
python sticky_snippet_generator.py 2500 0.2 5 5sticky.txt
python sticky_snippet_generator.py 2500 0.2 6 6sticky.txt
python sticky_snippet_generator.py 2500 0.2 7 7sticky.txt
python sticky_snippet_generator.py 2500 0.2 8 8sticky.txt
python sticky_snippet_generator.py 5000 0.2 20 stickypal.txt

paste -d '\n' 1sticky.txt 2sticky.txt > sticky12.txt
paste -d '\n' 3sticky.txt 4sticky.txt > sticky34.txt
paste -d '\n' 5sticky.txt 6sticky.txt > sticky56.txt
paste -d '\n' 7sticky.txt 8sticky.txt > sticky78.txt
rm 1sticky.txt
rm 2sticky.txt
rm 3sticky.txt
rm 4sticky.txt
rm 5sticky.txt
rm 6sticky.txt
rm 7sticky.txt
rm 8sticky.txt

mv *.txt test_5k_2

# create 5k 0.4 test folder
mkdir -m777 test_5k_4
python sticky_snippet_generator.py 5000 0.4 0 nosticky.txt
python sticky_snippet_generator.py 2500 0.4 1 1sticky.txt
python sticky_snippet_generator.py 2500 0.4 2 2sticky.txt
python sticky_snippet_generator.py 2500 0.4 3 3sticky.txt
python sticky_snippet_generator.py 2500 0.4 4 4sticky.txt
python sticky_snippet_generator.py 2500 0.4 5 5sticky.txt
python sticky_snippet_generator.py 2500 0.4 6 6sticky.txt
python sticky_snippet_generator.py 2500 0.4 7 7sticky.txt
python sticky_snippet_generator.py 2500 0.4 8 8sticky.txt
python sticky_snippet_generator.py 5000 0.4 20 stickypal.txt

paste -d '\n' 1sticky.txt 2sticky.txt > sticky12.txt
paste -d '\n' 3sticky.txt 4sticky.txt > sticky34.txt
paste -d '\n' 5sticky.txt 6sticky.txt > sticky56.txt
paste -d '\n' 7sticky.txt 8sticky.txt > sticky78.txt
rm 1sticky.txt
rm 2sticky.txt
rm 3sticky.txt
rm 4sticky.txt
rm 5sticky.txt
rm 6sticky.txt
rm 7sticky.txt
rm 8sticky.txt

mv *.txt test_5k_4

# create 5k 0.6 test folder
mkdir -m777 test_5k_6
python sticky_snippet_generator.py 5000 0.6 0 nosticky.txt
python sticky_snippet_generator.py 2500 0.6 1 1sticky.txt
python sticky_snippet_generator.py 2500 0.6 2 2sticky.txt
python sticky_snippet_generator.py 2500 0.6 3 3sticky.txt
python sticky_snippet_generator.py 2500 0.6 4 4sticky.txt
python sticky_snippet_generator.py 2500 0.6 5 5sticky.txt
python sticky_snippet_generator.py 2500 0.6 6 6sticky.txt
python sticky_snippet_generator.py 2500 0.6 7 7sticky.txt
python sticky_snippet_generator.py 2500 0.6 8 8sticky.txt
python sticky_snippet_generator.py 5000 0.6 20 stickypal.txt

paste -d '\n' 1sticky.txt 2sticky.txt > sticky12.txt
paste -d '\n' 3sticky.txt 4sticky.txt > sticky34.txt
paste -d '\n' 5sticky.txt 6sticky.txt > sticky56.txt
paste -d '\n' 7sticky.txt 8sticky.txt > sticky78.txt
rm 1sticky.txt
rm 2sticky.txt
rm 3sticky.txt
rm 4sticky.txt
rm 5sticky.txt
rm 6sticky.txt
rm 7sticky.txt
rm 8sticky.txt

mv *.txt test_5k_6

