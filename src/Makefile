SRCTOP = .

CXX = g++ 

INCLUDES += -I$(SRCTOP) -I../include
LIBS +=

CXXFLAG = -std=c++11 -O3 -march=native -mfma -funroll-loops -ffast-math -Wno-unused-value -Wall -g -fopenmp

all:annexml

annexml: annexml.cc AnnexML.o LLEmbedding.o FileReader.o SphericalClustering.o NGT.o
	$(CXX) -o $@ $^ $(CXXFLAG) $(INCLUDES) $(LIBS)

sleec-predict: annexml.cc AnnexML.o FileReader.o SphericalClustering.o NGT.o
	$(CXX) $(INCLUDES) -c LLEmbedding.cc -o LLEmbedding-SLEEC.o $(CXXFLAG) -DSLEEC
	$(CXX) -o $@ $^ LLEmbedding-SLEEC.o $(CXXFLAG) $(INCLUDES) $(LIBS)

.cc.o:
	$(CXX) $(INCLUDES) -c $< $(CXXFLAG)

clean:
	rm -f *~ *.o annexml sleec-predict

