train:
	python3.5 train.py /home/chunmun/fyp/vardec/all.vardec --iterations 100 --learning-rate 0.1 --hidden 200 --num-features 200

oldtrain:
	# No L2, allvardec, crtrans
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --validation-filename /home/chunmun/fyp/crtranscript.txt.proc 2>&1| tee stdout1
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --fixed-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/crtranscript.txt.proc 2>&1| tee stdout2
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --learn-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/crtranscript.txt.proc 2>&1| tee stdout3
	# Low L2 0.0001, allvardec, crtrans
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0.0001 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --validation-filename /home/chunmun/fyp/crtranscript.txt.proc 2>&1| tee stdout4
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0.0001 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --fixed-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/crtranscript.txt.proc 2>&1| tee stdout5
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0.0001 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --learn-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/crtranscript.txt.proc 2>&1| tee stdout6
	# High L2 0.1, allvardec, crtrans
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0.1 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --validation-filename /home/chunmun/fyp/crtranscript.txt.proc 2>&1| tee stdout7
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0.1 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --fixed-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/crtranscript.txt.proc 2>&1| tee stdout8
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0.1 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --learn-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/crtranscript.txt.proc 2>&1| tee stdout9
	# No L2, allvardec, atrans
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --validation-filename /home/chunmun/fyp/atranscript.txt.proc 2>&1| tee stdout10
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --fixed-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/atranscript.txt.proc 2>&1| tee stdout11
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --learn-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/atranscript.txt.proc 2>&1| tee stdout12
	# Low L2 0.0001, allvardec, atrans
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0.0001 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --validation-filename /home/chunmun/fyp/atranscript.txt.proc 2>&1| tee stdout13
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0.0001 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --fixed-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/atranscript.txt.proc 2>&1| tee stdout14
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0.0001 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --learn-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/atranscript.txt.proc 2>&1| tee stdout15
	# High L2 0.1, allvardec, atrans
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0.1 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --validation-filename /home/chunmun/fyp/atranscript.txt.proc 2>&1| tee stdout16
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0.1 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --fixed-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/atranscript.txt.proc 2>&1| tee stdout17
	python3.5 train-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0.1 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --learn-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/atranscript.txt.proc 2>&1| tee stdout18

hugetrain:
	# No L2, huge, atrans
	python3.5 -u train-lstm.py /home/chunmun/fyp/huge --iterations 100 --l2 0 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --validation-filename /home/chunmun/fyp/atranscript.txt.proc 2>&1| tee stdout19
	python3.5 -u train-lstm.py /home/chunmun/fyp/huge --iterations 100 --l2 0 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --fixed-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/atranscript.txt.proc 2>&1| tee stdout20
	python3.5 -u train-lstm.py /home/chunmun/fyp/huge --iterations 100 --l2 0 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --learn-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/atranscript.txt.proc 2>&1| tee stdout21

test:
	python3.5 -u test-lstm.py /home/chunmun/fyp/vardec/all.vardec --iterations 150 --l2 0.1 --learning-rate 1 --hidden 256 --num-features 50 --window 1 --learn-embeddings /home/chunmun/fyp/glove.6B.50d.txt --validation-filename /home/chunmun/fyp/atranscript.txt.proc --load-models 2>&1| tee stdout200

old:
	python3.5 -u train-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 200 --l2 1.0 --learning-rate 1 --hidden 256 --num-features 50 --dropout-rare 0.0 --validation-filename /home/chunmun/git/theanofun/data/crtranscript.txt.proc 2>&1| tee stdout20
	python3.5 -u train-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 200 --l2 1.0 --learning-rate 1 --hidden 256 --num-features 50 --dropout-rare 0.2 --validation-filename /home/chunmun/git/theanofun/data/crtranscript.txt.proc 2>&1| tee stdout21
	python3.5 -u train-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 200 --l2 1.0 --learning-rate 1 --hidden 256 --num-features 50 --dropout-rare 0.5 --validation-filename /home/chunmun/git/theanofun/data/crtranscript.txt.proc 2>&1| tee stdout22
	python3.5 -u train-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 200 --l2 1.0 --learning-rate 1 --hidden 256 --num-features 50 --dropout-rare 0.7 --validation-filename /home/chunmun/git/theanofun/data/crtranscript.txt.proc 2>&1| tee stdout23
	python3.5 -u train-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 200 --l2 1.0 --learning-rate 1 --hidden 256 --num-features 50 --dropout-rare 0.2 --validation-filename /home/chunmun/git/theanofun/data/atranscript.txt.proc 2>&1| tee stdout31
	python3.5 -u train-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 200 --l2 1.0 --learning-rate 1 --hidden 256 --num-features 50 --dropout-rare 0.5 --validation-filename /home/chunmun/git/theanofun/data/atranscript.txt.proc 2>&1| tee stdout32
	python3.5 -u train-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 200 --l2 1.0 --learning-rate 1 --hidden 256 --num-features 50 --dropout-rare 0.0 --validation-filename /home/chunmun/git/theanofun/data/atranscript.txt.proc 2>&1| tee stdout30
	python3.5 -u train-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 200 --l2 1.0 --learning-rate 1 --hidden 256 --num-features 50 --dropout-rare 0.7 --validation-filename /home/chunmun/git/theanofun/data/atranscript.txt.proc 2>&1| tee stdout33
	python3.5 -u train-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 200 --l2 0.01 --learning-rate 0.1 --hidden 256 --num-features 50 --dropout-rare 0.0 --validation-filename /home/chunmun/git/theanofun/data/atranscript.txt.proc 2>&1| tee stdout40
	python3.5 -u train-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 200 --l2 0.01 --learning-rate 0.1 --hidden 256 --num-features 50 --dropout-rare 0.2 --validation-filename /home/chunmun/git/theanofun/data/atranscript.txt.proc 2>&1| tee stdout41
	python3.5 -u train-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 200 --l2 0.01 --learning-rate 0.1 --hidden 256 --num-features 50 --dropout-rare 0.7 --validation-filename /home/chunmun/git/theanofun/data/atranscript.txt.proc 2>&1| tee stdout43

lstmtrain:
	python3.5 -u train-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 200 --l2 0.01 --learning-rate 0.1 --hidden 256 --num-features 50 --dropout-rare 0.5 --load-models --validation-filename /home/chunmun/git/theanofun/data/atranscript.txt.proc 2>&1| tee stdout42


tag:
	python3.5 -u tag-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 200 --l2 0.01 --learning-rate 0.1 --hidden 256 --num-features 50 --dropout-rare 0.5 --validation-filename /home/chunmun/git/theanofun/data/atranscript.txt.proc
	#python3.5 -u tag-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 150 --l2 0.1 --learning-rate 1 --hidden 256 --num-features 50 --dropout-rare 0.2 --validation-filename /home/chunmun/git/theanofun/data/crtranscript.txt.proc
	#python3.5 -u tag-lstm.py /home/chunmun/git/theanofun/data/all.vardec --iterations 150 --l2 0.1 --learning-rate 1 --hidden 256 --num-features 50 --dropout-rare 0.1 --dropout 0.5 --validation-filename /home/chunmun/git/theanofun/data/crtranscript.txt.proc
	#python3.5 tag-lstm.py --iterations 100 --hidden 32 --num-features 10 --window 1
	#python3.5 tag-lstm.py --iterations 100 --hidden 256 --num-features 50 --window 1
	#python3.5 tag.py --iterations 100 --learning-rate 0.1 --hidden 200 --num-features 200
