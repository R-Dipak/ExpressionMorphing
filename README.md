# ExpressionMorphing

An application which morphs any expression on a neutral face, based on input expression.

(NeutralFace1 , ExpressionFace1) + NeutralFace2  --> ExpressionFace2

Source for images : Srcforimages.cpp

Source for videos : drivervid.cpp

To run,

$ mkdir build

$ cd build

$ cmake ..

$ make

$ ./Srcforimages shape_predictor_68_face_landmarks.dat <NeutralFace1> <ExpressionFace1> <NeutralFace2>

