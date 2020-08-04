# To do
- [X] get the path
- [X] using patch-based pipelines
- [X] read about multi-classification
- [X] test if the metrics works
- [X] do this with another subject list (Or do I really need this? a separate list of path might be more helpful actually)
- [X] why the log part do not work
- [X] rewrite the store log part
- [X] why the model can not learn??? some part must be wrong. (Resample + cropped): it does a litte bit better, but it still happen again
- [X] try again not include the background
- [ ] why the dice score is always zero???
- [ ] might need more place to overlap with each other(could be a place to dip into) (how to remove borders) (look for the border pharams) (how the torchio deal with the overlapping part?)
- [ ] fix the bug about torchio (???)
- [ ] add the one more layer
- [ ] read about the learning rate
- [ ] why the matrix is different from the loss??
- [ ] why the print network show so less information now?
- [ ] crop and resample the data
- [ ] tune the patch based parameters to make the train faster
- [ ] use cropped image to train
- [ ] remember to find the suitable patch size
- [ ] make a parameter of whether to compute the background
- [ ] to test the dice loss
- [ ] add more Fully Connected Layer(???)
- [ ] rewrite the test part
- [ ] add one more layer without using concat
- [ ] force to upload once to github to clear some redundant file



## Try
- [ ] using leave one out validation


# visualization part
- [X] find out why _Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers)_. (if still cannot fix, we can go along with the gray image)
- [X] find out the reason why the second row of visualization do not working
- [X] make the data plot according to the original scale
- [X] why the color part isn't right????
- [ ] add mp4 part in the validation step end later
- [ ] adding the name of each axs


## Some code that might be useful
- in utils, there is a squeeze_data.py file which is used for squeeze
the label data (in ADNI, label data have an extra dimension like this (1, 192, ))

# The elements of networks
- loss function and matrixs: all compute the background 
- the visualization part:
    0. Using the whole brain volume to do the visualization part
    1. randomly choose 150 images (but every time the image is fixed) from the 1069 baseline images in the ADNI dataset
    2. using this images to do visualization everytime (with the same order every time)

# Some Performance
- if not using resample, using the orginal dice loss ( reduce by mean ), the loss would drop at a special moment to around 0.4, and all predict the background class. 
- but after resample, the model can learn. 


![](./img/brain_parcellation.png)