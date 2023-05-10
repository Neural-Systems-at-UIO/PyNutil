# related to object_counting: labelPoints
def labelPoints(points, label_volume, scale_factor=1):
    """this function takes a list of points and assigns them to a region based on the regionVolume.
    These regions will just be the values in the regionVolume at the points.
    it returns a dictionary with the region as the key and the points as the value"""
    #first convert the points to 3 columns
    points = np.reshape(points, (-1,3))
    #scale the points
    points = points * scale_factor
    #round the points to the nearest whole number
    points = np.round(points).astype(int)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    #get the label value for each point
    labels = label_volume[x,y,z]
    return labels