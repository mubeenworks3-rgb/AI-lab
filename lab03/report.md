# Laboratory 3 Report

## Part A: Text Features

1. Each column in the document-term matrix represents one vocabulary term. Each row is a document, and a cell contains that term's count in that document.
2. The machine-understandable representation is a fixed-length numeric vector of token counts for each document, arranged as a document-term matrix.
3. Bag-of-Words discards word order and much of context. It does not understand semantics, so synonyms are separate features; phrases such as `not good` do not encode negation as a relationship.

## Part B: Image Features

4. Grayscale conversion maps each pixel's red, green, and blue values to one luminance value. This reduces the number of features from three per pixel to one, while retaining brightness and spatial layout but discarding colour.
5. An image of height $H$ and width $W$, after grayscale conversion, has $H \times W$ values. Its flattened vector therefore has length $HW$.
6. Flattening places the pixels into a one-dimensional vector, which is the input shape expected by basic estimators such as Logistic Regression and Support Vector Machines. It makes all samples compatible with a standard feature matrix, although it does not preserve explicit two-dimensional neighbourhood structure for the model.
7. Resizing every image to the same dimensions guarantees that all feature vectors have the same length. For example, a $64 \times 64$ grayscale image becomes a vector of length 4,096, allowing the images to be stacked into one matrix.

## Home Assignment: 20 Samples in 4,096 Dimensions

The dataset has 20 samples and 4,096 pixel features per sample, so its feature matrix has shape $(20, 4096)$. This is a very small dataset relative to the number of dimensions: a traditional model can easily overfit, and the samples may not represent the variation within each class. More labelled images, dimensionality reduction, regularisation, or a model that learns useful spatial features would provide a stronger basis for generalisation.
