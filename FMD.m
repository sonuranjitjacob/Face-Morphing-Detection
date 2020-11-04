clear all;
close all;
images = imageSet('Cropped\MorphedImages\Train');
real = imageSet('Cropped\RealImages\Train');
realtest = imageSet('Cropped\RealImages\Test');
morphedtest = imageSet('Cropped\MorphedImages\Test');

%%
cellSize = [4 4];
% hogFeatureSize = length(hog_4x4);
for j = 1:20
    img = read(images, j);
    FaceDetect = vision.CascadeObjectDetector;
    FaceDetect.MergeThreshold = 7 ;
    BB = step(FaceDetect, img);
    for i = 1 : size(BB,1)
        rectangle('Position', BB(i,:), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'r');
    end
    for i = 1 : size(BB, 1)
        J = imcrop(img, BB(i, :));
    end
    J_gray = imresize(J, [865 865]);
    J_gray = rgb2gray(J_gray);
    J_gray = imbinarize(J_gray);
    trainingFeatures(j,:) = extractHOGFeatures(J_gray);
end

%%

for j = 21:30
    img = read(real, j);
    FaceDetect = vision.CascadeObjectDetector;
    FaceDetect.MergeThreshold = 7 ;
    BB = step(FaceDetect, img);
    for i = 1 : size(BB,1)
        rectangle('Position', BB(i,:), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'r');
    end
    for i = 1 : size(BB, 1)
        J = imcrop(img, BB(i, :));
    end
    J_gray = imresize(J, [865 865]);
    J_gray = rgb2gray(J_gray);
    J_gray = imbinarize(J_gray);
    trainingFeatures(j,:) = extractHOGFeatures(J_gray);
end
%%
trainingLabels = zeros(20,1);
train_new = ones(10, 1);
trainingLabels = [trainingLabels;train_new];

%%
classifier = fitcecoc(trainingFeatures, trainingLabels);

%%
for j = 1:20
    img = read(images, j);
    FaceDetect = vision.CascadeObjectDetector;
    FaceDetect.MergeThreshold = 7 ;
    BB = step(FaceDetect, img);
    for i = 1 : size(BB,1)
        rectangle('Position', BB(i,:), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'r');
    end
    for i = 1 : size(BB, 1)
        J = imcrop(img, BB(i, :));
    end
    J_gray = imresize(J, [865 865]);
    J_gray = rgb2gray(J_gray);
    J_gray = imbinarize(J_gray);
    testFeatures(j, :) = extractHOGFeatures(J_gray);
end

%%
for j = 21:30
    img = read(real, j);
    FaceDetect = vision.CascadeObjectDetector;
    FaceDetect.MergeThreshold = 7 ;
    BB = step(FaceDetect, img);
    for i = 1 : size(BB,1)
        rectangle('Position', BB(i,:), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'r');
    end
    for i = 1 : size(BB, 1)
        J = imcrop(img, BB(i, :));
    end
    J_gray = imresize(J, [865 865]);
    J_gray = rgb2gray(J_gray);
    J_gray = imbinarize(J_gray);
    testFeatures(j, :) = extractHOGFeatures(J_gray);
end
%%

predictedLabels = predict(classifier, testFeatures);




%% project old code

clear all;
close all;
images = imageSet('Cropped\MorphedImages\Train');
realimg = imageSet('Cropped\RealImages\Train');
realtest = imageSet('Cropped\RealImages\Test');
morphedtest = imageSet('Cropped\MorphedImages\Test');

%%
%%
cellSize = [4 4];
i
% hogFeatureSize = length(hog_4x4);
for j = 1:3040
    img = read(images, j);
    trainingFeatures(j,:) = extractHOGFeatures(img);
end

%%

for j = 21:30
    img = read(realimg, j);
    trainingFeatures(j,:) = extractHOGFeatures(img);
end
%%
trainingLabels = zeros(20,1);
train_new = ones(10, 1);
trainingLabels = [trainingLabels;train_new];

%%
classifier = fitcecoc(trainingFeatures, trainingLabels);

%%
for j = 1:20
    img = read(images, j);
    testFeatures(j, :) = extractHOGFeatures(img);
end

%%
for j = 21:30
    img = read(realimg, j);
    testFeatures(j, :) = extractHOGFeatures(img);
end
%%

predictedLabels = predict(classifier, testFeatures);