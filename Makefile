CXX = g++

CFLAGS = -O3 -I. -Wno-c++11-narrowing -Wno-unused-parameter -Wno-unused-function -std=c++11 `pkg-config --cflags opencv tesseract`

LIBS = `pkg-config --libs-only-L --libs-only-l opencv tesseract`

DEPS = group_classifier.h max_meaningful_clustering.h mser.h min_bounding_box.h region_classifier.h region.h text_extract.h utils.h FaceDetector.h TextDetector.h VideoReader.h ImageUtils.h

OBJ = fast_clustering.o group_classifier.o max_meaningful_clustering.o min_bounding_box.o mser.o nfa.o utils.o region_classifier.o region.o FaceDetector.o TextDetector.o VideoReader.o ImageUtils.o

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CFLAGS)

all: Mechanic OfflineProcessor

Mechanic: $(OBJ) main.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBS)
	
OfflineProcessor: $(OBJ) OfflineProcessor.cpp
	$(CXX) -o OfflineProcessor $^ $(CFLAGS) $(LIBS)
clean:
	rm -rf $(OBJ) Mechanic
