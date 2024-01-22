import os
import numpy as np
import pandas as pd
import imagesize
import matplotlib.pyplot as plt


class_names = {
    0:'Person',
    1:'Car',
    2:'Truck',
    3:'UAV',
    4:'Aircraft',
    5:'Ship'
}


def inspect_dataset(
        target_dataset_root,  # ../datasets/new_dataset
        target_dataset_slice,  # train,test,val
):
 info = []  # list of lists, each list corresponds to an instance [cls_id, x, y, w, h, img]

 target_labels_dir = os.path.join(target_dataset_root, 'labels', target_dataset_slice)

 # Iterate over all files in the original dataset labels folder
 for filename in os.listdir(target_labels_dir):
  if filename.endswith('.txt'):
   # Read file
   with open(os.path.join(target_labels_dir, filename), "r") as f:
    # Iterate over instances in image and get present class ids
    for line in f:
     line_data = []
     # label data
     line_data = line.split()
     # Image name
     line_data.append(os.path.splitext(filename)[0])
     # Image size: could be done at image level and not row level
     img_path = os.path.join(target_dataset_root, 'images', target_dataset_slice,
                             os.path.splitext(filename)[0] + '.jpg')
     img_w, img_h = imagesize.get(img_path)
     line_data.extend([img_w, img_h])
     # Append line data to info
     info.append(line_data)

 df = pd.DataFrame(info, columns=['new_class_id', 'xcn', 'ycn', 'wn', 'hn', 'img', 'img_w', 'img_h'])
 df = df.astype(
  {'new_class_id': 'int32', 'xcn': 'float32', 'ycn': 'float32', 'wn': 'float32', 'hn': 'float32', 'img': 'int64',
   'img_w': 'float32', 'img_h': 'float32'})
 df['class_name'] = df['new_class_id'].map(class_names)
 return df

df_val = inspect_dataset('/data-fast/127-data2/ierregue/datasets/custom_dataset_v2', 'val')
df_train = inspect_dataset('/data-fast/127-data2/ierregue/datasets/custom_dataset_v2', 'train')

print(f"The number of objects is {len(df_val)}")
print(f"The number of images is {len(df_val['img'].unique())}")

print(f"The number of objects is {len(df_train)}")
print(f"The number of images is {len(df_train['img'].unique())}")

# Create dir to store plots
save_dir = './data/dataset_creation/plots/v2'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

plt.rcParams.update({'font.size': 10})
plt.rcParams['axes.axisbelow'] = True

fig = plt.figure(figsize=(3, 3))
ax = df_train['class_name'].value_counts().plot(kind='bar', width=0.75, zorder=3, label='Training')
# Set x-axis label
ax.set_xlabel("Classes", weight='bold', size=12)
# Set y-axis label
ax.set_ylabel("Counts", weight='bold', size=12)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, size=8)
ax.grid()
#plt.legend()
None
fig.savefig(save_dir+'/train_class_counts.png', bbox_inches = 'tight')


fig = plt.figure(figsize=(3, 3))
ax = df_val['class_name'].value_counts().plot(kind='bar', width=0.75, zorder=3, color='tab:red', label='Validation')
# Set x-axis label
ax.set_xlabel("Classes", weight='bold', size=12)
# Set y-axis label
ax.set_ylabel("Counts", weight='bold', size=12)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, size=8)
ax.grid()
plt.legend()
None
fig.savefig(save_dir+f'/val_class_counts.png', bbox_inches = 'tight')


"""fig = plt.figure(figsize=(6.2, 3))
ax = df_train.groupby(by=['img'])['img'].count().value_counts().sort_index().plot(kind='bar', zorder=3, label='Training')
# Set x-axis label
ax.set_xlabel("Instances per image", weight='bold', size=12)
# Set y-axis label
ax.set_ylabel("Counts", weight='bold', size=12)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, size=7)
ax.grid()
#plt.legend()
None
fig.savefig(save_dir+f'/train_instances_count.png', bbox_inches = 'tight')


fig = plt.figure(figsize=(6.2, 3))
ax = df_val.groupby(by=['img'])['img'].count().value_counts().sort_index().plot(kind='bar', zorder=3, color='tab:red', label='Validation')
# Set x-axis label
ax.set_xlabel("Instances per image", weight='bold', size=12)
# Set y-axis label
ax.set_ylabel("Counts", weight='bold', size=12)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, size=7)
ax.grid()
#plt.legend()
None
fig.savefig(save_dir+f'/val_instances_count.png', bbox_inches = 'tight')"""

data = df_train['img'].value_counts()
bins = [1,4,7,10,16,data.max()+1]
hist, bin_edges = np.histogram(data,bins) # make the histogram
fig,ax = plt.subplots(figsize=(3, 3))
# Plot the histogram heights against integers on the x axis
ax.bar(range(len(hist)),hist,width=0.9)
# Set the ticks to the middle of the bars
ax.set_xticks([i for i,j in enumerate(hist)])
# Set the xticklabels to a string that tells us what the bin edges were
ax.set_xticklabels(['{}-{}'.format(bins[i],bins[i+1]-1) for i,j in enumerate(hist)], size=8)
ax.set_xlabel("Instances per image", weight='bold', size=12)
ax.set_ylabel("Counts", weight='bold', size=12)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
ax.grid()
None
fig.savefig(save_dir + '/train_instances_count.png', bbox_inches = 'tight')

data = df_val['img'].value_counts()
bins = [1,4,7,10,16,data.max()+1]
hist, bin_edges = np.histogram(data,bins) # make the histogram
fig,ax = plt.subplots(figsize=(3, 3))
# Plot the histogram heights against integers on the x axis
ax.bar(range(len(hist)),hist,width=0.9, color='tab:red')
# Set the ticks to the middle of the bars
ax.set_xticks([i for i,j in enumerate(hist)])
# Set the xticklabels to a string that tells us what the bin edges were
ax.set_xticklabels(['{}-{}'.format(bins[i],bins[i+1]-1) for i,j in enumerate(hist)], size=8)
ax.set_xlabel("Instances per image", weight='bold', size=12)
ax.set_ylabel("Counts", weight='bold', size=12)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
ax.grid()
None
fig.savefig(save_dir + '/val_instances_count.png', bbox_inches = 'tight')


df_train['bbox_area'] = (df_train['wn']*df_train['img_w'])*(df_train['hn']*df_train['img_h'])
#df_train['bbox_image_area_ration'] = df_train['bbox_area']/(df_train['img_w']*df_train['img_h'])

#bin_edges = [0, 1/1200, 1/300, 3/100, float('inf')]
#bin_labels = ['Tiny', 'Small', 'Medium', 'Large']
#df_train['bbox_size_category'] = pd.cut(df_train['bbox_image_area_ration'], bins=bin_edges, labels=bin_labels, right=False)

bin_edges = [0, 16**2, 32**2, 96**2, float('inf')]
bin_labels = ['Tiny', 'Small', 'Medium', 'Large']
df_train['bbox_size_category'] = pd.cut(df_train['bbox_area'], bins=bin_edges, labels=bin_labels, right=False)

df_val['bbox_area'] = (df_val['wn']*df_val['img_w'])*(df_val['hn']*df_val['img_h'])
df_val['bbox_image_area_ration'] = df_val['bbox_area']/(df_val['img_w']*df_val['img_h'])
#df_val['bbox_size_category'] = pd.cut(df_val['bbox_image_area_ration'], bins=bin_edges, labels=bin_labels, right=False)
df_val['bbox_size_category'] = pd.cut(df_val['bbox_area'], bins=bin_edges, labels=bin_labels, right=False)


fig = plt.figure(figsize=(3, 3))
ax = df_train['bbox_size_category'].value_counts().sort_index().plot(kind='bar', width=0.7, zorder=3, label='Training')
# Set x-axis label
ax.set_xlabel("Object category size", weight='bold', size=12)
# Set y-axis label
ax.set_ylabel("Counts", weight='bold', size=12)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid()
#plt.legend()
None
fig.savefig(save_dir+f'/train_objsz_counts.png', bbox_inches = 'tight')

fig = plt.figure(figsize=(3, 3))
ax = df_val['bbox_size_category'].value_counts().sort_index().plot(kind='bar', width=0.7, zorder=3, color='tab:red',label='Validation')
# Set x-axis label
ax.set_xlabel("Object category size", weight='bold', size=12)
# Set y-axis label
ax.set_ylabel("Counts", weight='bold', size=12)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid()
#plt.legend()
None
fig.savefig(save_dir+f'/val_objsz_counts.png', bbox_inches = 'tight')