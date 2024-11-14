#!/usr/bin/env python
# coding: utf-8

# # Image Classification

# In[ ]:


# This block is needed (only) for Colab
# If Colab: Click Runtime -> Change runtime type -> select Hardware accelerator: GPU
# !pip install -Uqq fastbook
# import fastbook
# fastbook.setup_book()


# ## Classfication of Pet Breeds 

# In[1]:


from fastai.vision.all import *


# ### Data Preparation

# In[ ]:


path = untar_data(URLs.PETS)


# ### Preparing the FastAi Datablock

# In[ ]:


pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))


# ### Loading the Data

# In[ ]:


dls = pets.dataloaders(path/"images", num_workers=0)


# ### Checking if Data and Labeling is Correct

# In[ ]:


dls.show_batch(nrows=3, ncols=3)


# ### Define the Parameters of the Model

# In[ ]:


learn = vision_learner(dls, resnet34, metrics=error_rate)


# ### Train the Model

# In[ ]:


learn.fine_tune(3)


# ### Model Interpretation

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=80)


# In[ ]:


interp.most_confused(min_val=5)


# In[ ]:




