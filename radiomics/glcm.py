import numpy
from six.moves import range

from radiomics import base, cMatrices, cMatsEnabled, deprecated, imageoperations


class RadiomicsGLCM(base.RadiomicsFeaturesBase):
  r"""
  A Gray Level Co-occurrence Matrix (GLCM) of size :math:`N_g \times N_g` describes the second-order joint probability
  function of an image region constrained by the mask and is defined as :math:`\textbf{P}(i,j|\delta,\theta)`.
  The :math:`(i,j)^{\text{th}}` element of this matrix represents the number of times the combination of
  levels :math:`i` and :math:`j` occur in two pixels in the image, that are separated by a distance of :math:`\delta`
  pixels along angle :math:`\theta`.
  The distance :math:`\delta` from the center voxel is defined as the distance according to the infinity norm.
  For :math:`\delta=1`, this results in 2 neighbors for each of 13 angles in 3D (26-connectivity) and for
  :math:`\delta=2` a 98-connectivity (49 unique angles).

  Note that pyradiomics by default computes symmetrical GLCM!

  As a two dimensional example, let the following matrix :math:`\textbf{I}` represent a 5x5 image, having 5 discrete
  grey levels:

  .. math::
    \textbf{I} = \begin{bmatrix}
    1 & 2 & 5 & 2 & 3\\
    3 & 2 & 1 & 3 & 1\\
    1 & 3 & 5 & 5 & 2\\
    1 & 1 & 1 & 1 & 2\\
    1 & 2 & 4 & 3 & 5 \end{bmatrix}

  For distance :math:`\delta = 1` (considering pixels with a distance of 1 pixel from each other)
  and angle :math:`\theta=0^\circ` (horizontal plane, i.e. voxels to the left and right of the center voxel),
  the following symmetrical GLCM is obtained:

  .. math::
    \textbf{P} = \begin{bmatrix}
    6 & 4 & 3 & 0 & 0\\
    4 & 0 & 2 & 1 & 3\\
    3 & 2 & 0 & 1 & 2\\
    0 & 1 & 1 & 0 & 0\\
    0 & 3 & 2 & 0 & 2 \end{bmatrix}

  Let:

  - :math:`\epsilon` be an arbitrarily small positive number (:math:`\approx 2.2\times10^{-16}`)
  - :math:`\textbf{P}(i,j)` be the co-occurence matrix for an arbitrary :math:`\delta` and :math:`\theta`
  - :math:`p(i,j)` be the normalized co-occurence matrix and equal to
    :math:`\frac{\textbf{P}(i,j)}{\sum{\textbf{P}(i,j)}}`
  - :math:`N_g` be the number of discrete intensity levels in the image
  - :math:`p_x(i) = \sum^{N_g}_{j=1}{P(i,j)}` be the marginal row probabilities
  - :math:`p_y(j) = \sum^{N_g}_{i=1}{P(i,j)}` be the marginal column probabilities
  - :math:`\mu_x` be the mean gray level intensity of :math:`p_x` and defined as
    :math:`\mu_x = \displaystyle\sum^{N_g}_{i=1}{p_x(i)i}`
  - :math:`\mu_y` be the mean gray level intensity of :math:`p_y` and defined as
    :math:`\mu_y = \displaystyle\sum^{N_g}_{j=1}{p_y(j)j}`
  - :math:`\sigma_x` be the standard deviation of :math:`p_x`
  - :math:`\sigma_y` be the standard deviation of :math:`p_y`
  - :math:`p_{x+y}(k) = \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p(i,j)},\text{ where }i+j=k,\text{ and }k=2,3,\dots,2N_g`
  - :math:`p_{x-y}(k) = \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p(i,j)},\text{ where }|i-j|=k,\text{ and }k=0,1,\dots,N_g-1`
  - :math:`HX =  -\sum^{N_g}_{i=1}{p_x(i)\log_2\big(p_x(i)+\epsilon\big)}` be the entropy of :math:`p_x`
  - :math:`HY =  -\sum^{N_g}_{j=1}{p_y(j)\log_2\big(p_y(j)+\epsilon\big)}` be the entropy of :math:`p_y`
  - :math:`HXY =  -\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p(i,j)\log_2\big(p(i,j)+\epsilon\big)}` be the entropy of
    :math:`p(i,j)`
  - :math:`HXY1 =  -\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p(i,j)\log_2\big(p_x(i)p_y(j)+\epsilon\big)}`
  - :math:`HXY2 =  -\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_x(i)p_y(j)\log_2\big(p_x(i)p_y(j)+\epsilon\big)}`

  By default, the value of a feature is calculated on the GLCM for each angle separately, after which the mean of these
  values is returned. If distance weighting is enabled, GLCM matrices are weighted by weighting factor W and
  then summed and normalised. Features are then calculated on the resultant matrix.
  Weighting factor W is calculated for the distance between neighbouring voxels by:

  :math:`W = e^{-\|d\|^2}`, where d is the distance for the associated angle according
  to the norm specified in setting 'weightingNorm'.

  The following class specific settings are possible:

  - distances [[1]]: List of integers. This specifies the distances between the center voxel and the neighbor, for which
    angles should be generated. See also :py:func:`~radiomics.imageoperations.generateAngles`
  - symmetricalGLCM [True]: boolean, indicates whether co-occurrences should be assessed in two directions per angle,
    which results in a symmetrical matrix, with equal distributions for :math:`i` and :math:`j`. A symmetrical matrix
    corresponds to the GLCM as defined by Haralick et al.
  - weightingNorm [None]: string, indicates which norm should be used when applying distance weighting.
    Enumerated setting, possible values:

    - 'manhattan': first order norm
    - 'euclidean': second order norm
    - 'infinity': infinity norm.
    - 'no_weighting': GLCMs are weighted by factor 1 and summed
    - None: Applies no weighting, mean of values calculated on separate matrices is returned.

    In case of other values, an warning is logged and option 'no_weighting' is used.

  References

  - Haralick, R., Shanmugan, K., Dinstein, I; Textural features for image classification;
    IEEE Transactions on Systems, Man and Cybernetics; 1973(3), p610-621
  - `<https://en.wikipedia.org/wiki/Co-occurrence_matrix>`_
  - `<http://www.fp.ucalgary.ca/mhallbey/the_glcm.htm>`_
  """

  def __init__(self, inputImage, inputMask, **kwargs):
    super(RadiomicsGLCM, self).__init__(inputImage, inputMask, **kwargs)

    self.symmetricalGLCM = kwargs.get('symmetricalGLCM', True)
    self.weightingNorm = kwargs.get('weightingNorm', None)  # manhattan, euclidean, infinity

    self.P_glcm = None

    self._initSegmentBasedCalculation()

  def _initSegmentBasedCalculation(self):
    super(RadiomicsGLCM, self)._initSegmentBasedCalculation()

    self._applyBinning()

    if cMatsEnabled():
      self.P_glcm = self._calculateCMatrix()
    else:
      self.P_glcm = self._calculateMatrix()

    self._calculateCoefficients()

    self.logger.debug('GLCM feature class initialized, calculated GLCM with shape %s', self.P_glcm.shape)

  def _calculateMatrix(self):
    r"""
    Compute GLCMs for the input image for every direction in 3D.
    Calculated GLCMs are placed in array P_glcm with shape (i/j, a)
    i/j = total gray-level bins for image array,
    a = directions in 3D (generated by imageoperations.generateAngles)
    """
    self.logger.debug('Calculating GLCM matrix in Python')

    # Exclude voxels outside segmentation, due to binning, no negative values will be encountered inside the mask
    self.matrix[self.maskArray == 0] = -1

    angles = imageoperations.generateAngles(self.boundingBoxSize, **self.kwargs)

    grayLevels = self.coefficients['grayLevels']

    P_glcm = numpy.zeros((len(grayLevels), len(grayLevels), int(angles.shape[0])), dtype='float64')

    # If verbosity > INFO, or no progress reporter is set in radiomics.progressReporter, _dummyProgressReporter is used,
    # which just iterates over the iterator without reporting progress
    with self.progressReporter(grayLevels, desc='calculate GLCM') as bar:
      # iterate over gray levels for center voxel
      for i_idx, i in enumerate(bar):
        # get the indices to all voxels which have the current gray level i
        i_indices = numpy.where(self.matrix == i)

        # iterate over gray levels for neighbouring voxel
        for j_idx, j in enumerate(grayLevels):
          # get the indices to all voxels which have the current gray level j
          j_indices = set(zip(*numpy.where(self.matrix == j)))

          for a_idx, a in enumerate(angles):
            # get the corresponding indices of the neighbours for angle a
            neighbour_indices = set(zip(*(i_indices + a[:, None])))

            # The following intersection yields the indices to voxels with gray level j
            # that are also a neighbour of a voxel with gray level i for angle a.
            # The number of indices is then equal to the total number of pairs with gray level i and j for angle a
            count = len(neighbour_indices.intersection(j_indices))
            P_glcm[i_idx, j_idx, a_idx] = count

    P_glcm = self._applyMatrixOptions(P_glcm, angles)

    return P_glcm

  def _calculateCMatrix(self):
    self.logger.debug('Calculating GLCM matrix in C')

    angles = imageoperations.generateAngles(self.boundingBoxSize, **self.kwargs)
    Ng = self.coefficients['Ng']

    P_glcm = cMatrices.calculate_glcm(self.matrix, self.maskArray, angles, Ng)
    P_glcm = self._applyMatrixOptions(P_glcm, angles)

    # Delete rows and columns that specify gray levels not present in the ROI
    NgVector = range(1, Ng + 1)  # All possible gray values
    GrayLevels = self.coefficients['grayLevels']  # Gray values present in ROI
    emptyGrayLevels = numpy.array(list(set(NgVector) - set(GrayLevels)))  # Gray values NOT present in ROI

    P_glcm = numpy.delete(P_glcm, emptyGrayLevels - 1, 0)
    P_glcm = numpy.delete(P_glcm, emptyGrayLevels - 1, 1)

    return P_glcm

  def _applyMatrixOptions(self, P_glcm, angles):
    """
    Further process calculated matrix by optionally making it symmetrical and/or applying a weighting factor.
    Finally, delete empty angles and normalize the GLCM by dividing it by the sum of its elements.
    """
    self.logger.debug('Process calculated matrix')

    # Optionally make GLCMs symmetrical for each angle
    if self.symmetricalGLCM:
      self.logger.debug('Create symmetrical matrix')
      # Transpose and copy GLCM and add it to P_glcm. Numpy.transpose returns a view if possible, use .copy() to ensure
      # a copy of the array is used and not just a view (otherwise erroneous additions can occur)
      P_glcm += numpy.transpose(P_glcm, (1, 0, 2)).copy()

    # Optionally apply a weighting factor
    if self.weightingNorm is not None:
      self.logger.debug('Applying weighting (%s)', self.weightingNorm)
      pixelSpacing = self.inputImage.GetSpacing()[::-1]
      weights = numpy.empty(len(angles))
      for a_idx, a in enumerate(angles):
        if self.weightingNorm == 'infinity':
          weights[a_idx] = numpy.exp(-max(numpy.abs(a) * pixelSpacing) ** 2)
        elif self.weightingNorm == 'euclidean':
          weights[a_idx] = numpy.exp(-numpy.sum((numpy.abs(a) * pixelSpacing) ** 2))  # sqrt ^ 2 = 1
        elif self.weightingNorm == 'manhattan':
          weights[a_idx] = numpy.exp(-numpy.sum(numpy.abs(a) * pixelSpacing) ** 2)
        elif self.weightingNorm == 'no_weighting':
          weights[a_idx] = 1
        else:
          self.logger.warning('weigthing norm "%s" is unknown, W is set to 1', self.weightingNorm)
          weights[a_idx] = 1

      P_glcm = numpy.sum(P_glcm * weights[None, None, :], 2, keepdims=True)

    sumP_glcm = numpy.sum(P_glcm, (0, 1))

    # Delete empty angles if no weighting is applied
    if P_glcm.shape[2] > 1:
      emptyAngles = numpy.where(sumP_glcm == 0)
      if len(emptyAngles[0]) > 0:  # One or more angles are 'empty'
        self.logger.debug('Deleting %d empty angles:\n%s', len(emptyAngles[0]), angles[emptyAngles])
        P_glcm = numpy.delete(P_glcm, emptyAngles, 2)
        sumP_glcm = numpy.delete(sumP_glcm, emptyAngles, 0)
      else:
        self.logger.debug('No empty angles')

    # Normalize each glcm
    return P_glcm / sumP_glcm

  # check if ivector and jvector can be replaced
  def _calculateCoefficients(self):
    r"""
    Calculate and fill in the coefficients dict.
    """
    self.logger.debug('Calculating GLCM coefficients')

    Ng = self.coefficients['Ng']
    eps = numpy.spacing(1)

    NgVector = self.coefficients['grayLevels'].astype('float')
    # shape = (Ng, Ng)
    i, j = numpy.meshgrid(NgVector, NgVector, indexing='ij', sparse=True)

    # shape = (2*Ng-1)
    kValuesSum = numpy.arange(2, (Ng * 2) + 1)
    # shape = (Ng-1)
    kValuesDiff = numpy.arange(0, Ng)

    # marginal row probabilities #shape = (Ng, 1, angles)
    px = self.P_glcm.sum(1, keepdims=True)
    # marginal column probabilities #shape = (1, Ng, angles)
    py = self.P_glcm.sum(0, keepdims=True)

    # shape = (1, 1, angles)
    ux = numpy.sum(i[:, :, None] * self.P_glcm, (0, 1), keepdims=True)
    uy = numpy.sum(j[:, :, None] * self.P_glcm, (0, 1), keepdims=True)

    # shape = (1, 1, angles)
    sigx = numpy.sum(self.P_glcm * ((i[:, :, None] - ux) ** 2), (0, 1), keepdims=True) ** 0.5
    # shape = (1, 1, angles)
    sigy = numpy.sum(self.P_glcm * ((j[:, :, None] - uy) ** 2), (0, 1), keepdims=True) ** 0.5

    # shape = (2*Ng-1, angles)
    pxAddy = numpy.array([numpy.sum(self.P_glcm[i + j == k], 0) for k in kValuesSum])
    # shape = (Ng, angles)
    pxSuby = numpy.array([numpy.sum(self.P_glcm[numpy.abs(i - j) == k], 0) for k in kValuesDiff])

    # entropy of px # shape = (angles)
    HX = (-1) * numpy.sum((px * numpy.log2(px + eps)), (0, 1))
    # entropy of py # shape = (angles)
    HY = (-1) * numpy.sum((py * numpy.log2(py + eps)), (0, 1))
    # shape = (angles)
    HXY = (-1) * numpy.sum((self.P_glcm * numpy.log2(self.P_glcm + eps)), (0, 1))

    # shape = (angles)
    HXY1 = (-1) * numpy.sum((self.P_glcm * numpy.log2(px * py + eps)), (0, 1))
    # shape = (angles)
    HXY2 = (-1) * numpy.sum(((px * py) * numpy.log2(px * py + eps)), (0, 1))

    self.coefficients['eps'] = eps
    self.coefficients['i'] = i
    self.coefficients['j'] = j
    self.coefficients['kValuesSum'] = kValuesSum
    self.coefficients['kValuesDiff'] = kValuesDiff
    self.coefficients['px'] = px
    self.coefficients['py'] = py
    self.coefficients['ux'] = ux
    self.coefficients['uy'] = uy
    self.coefficients['sigx'] = sigx
    self.coefficients['sigy'] = sigy
    self.coefficients['pxAddy'] = pxAddy
    self.coefficients['pxSuby'] = pxSuby
    self.coefficients['HX'] = HX
    self.coefficients['HY'] = HY
    self.coefficients['HXY'] = HXY
    self.coefficients['HXY1'] = HXY1
    self.coefficients['HXY2'] = HXY2

  def getAutocorrelationFeatureValue(self):
    r"""
    **1. Autocorrelation**

    .. math::
      \textit{autocorrelation} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{p(i,j)ij}

    Autocorrelation is a measure of the magnitude of the fineness and coarseness of texture.
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    ac = numpy.sum(self.P_glcm * (i * j)[:, :, None], (0, 1))
    return ac.mean()

  def getJointAverageFeatureValue(self):
    r"""
    **2. Joint Average**

    .. math::
      \textit{joint average} = \mu_x = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{p(i,j)i}

    Returns the mean gray level intensity of the :math:`i` distribution.

    .. warning::
      As this formula represents the average of the distribution of :math:`i`, it is independent from the
      distribution of :math:`j`. Therefore, only use this formula if the GLCM is symmetrical, where
      :math:`p_x(i) = p_y(j) \text{, where } i = j`.
    """
    if not self.symmetricalGLCM:
      self.logger.warning('The formula for GLCM - Joint Average assumes that the GLCM is symmetrical, but this is not the case.')
    return self.coefficients['ux'].mean()

  def getClusterProminenceFeatureValue(self):
    r"""
    **3. Cluster Prominence**

    .. math::
      \textit{cluster prominence} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}
      {\big( i+j-\mu_x-\mu_y\big)^4p(i,j)}

    Cluster Prominence is a measure of the skewness and asymmetry of the GLCM. A higher values implies more asymmetry
    about the mean while a lower value indicates a peak near the mean value and less variation about the mean.
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    ux = self.coefficients['ux']
    uy = self.coefficients['uy']
    cp = numpy.sum((self.P_glcm * (((i + j)[:, :, None] - ux - uy) ** 4)), (0, 1))
    return cp.mean()

  def getClusterShadeFeatureValue(self):
    r"""
    **4. Cluster Shade**

    .. math::
      \textit{cluster shade} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}
      {\big(i+j-\mu_x-\mu_y\big)^3p(i,j)}

    Cluster Shade is a measure of the skewness and uniformity of the GLCM.
    A higher cluster shade implies greater asymmetry about the mean.
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    ux = self.coefficients['ux']
    uy = self.coefficients['uy']
    cs = numpy.sum((self.P_glcm * (((i + j)[:, :, None] - ux - uy) ** 3)), (0, 1))
    return cs.mean()

  def getClusterTendencyFeatureValue(self):
    r"""
    **5. Cluster Tendency**

    .. math::
      \textit{cluster tendency} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}
      {\big(i+j-\mu_x-\mu_y\big)^2p(i,j)}

    Cluster Tendency is a measure of groupings of voxels with similar gray-level values.
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    ux = self.coefficients['ux']
    uy = self.coefficients['uy']
    ct = numpy.sum((self.P_glcm * (((i + j)[:, :, None] - ux - uy) ** 2)), (0, 1))
    return ct.mean()

  def getContrastFeatureValue(self):
    r"""
    **6. Contrast**

    .. math::
      \textit{contrast} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{(i-j)^2p(i,j)}

    Contrast is a measure of the local intensity variation, favoring values away from the diagonal :math:`(i = j)`. A
    larger value correlates with a greater disparity in intensity values among neighboring voxels.
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    cont = numpy.sum((self.P_glcm * ((numpy.abs(i - j))[:, :, None] ** 2)), (0, 1))
    return cont.mean()

  def getCorrelationFeatureValue(self):
    r"""
    **7. Correlation**

    .. math::
      \textit{correlation} = \frac{\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p(i,j)ij-\mu_x\mu_y}}{\sigma_x(i)\sigma_y(j)}

    Correlation is a value between 0 (uncorrelated) and 1 (perfectly correlated) showing the
    linear dependency of gray level values to their respective voxels in the GLCM.

    .. note::
      When there is only 1 discreet gray value in the ROI (flat region), :math:`\sigma_x` and :math:`\sigma_y` will be
      0. In this case, an arbitrary value of 1 is returned instead. This is assessed on a per-angle basis.
    """
    eps = self.coefficients['eps']
    i = self.coefficients['i']
    j = self.coefficients['j']
    ux = self.coefficients['ux']
    uy = self.coefficients['uy']
    sigx = self.coefficients['sigx']
    sigy = self.coefficients['sigy']

    corm = numpy.sum(self.P_glcm * (i[:, :, None] - ux) * (j[:, :, None] - uy), (0, 1), keepdims=True)
    corr = corm / (sigx * sigy + eps)
    corr[sigx * sigy == 0] = 1  # Set elements that would be divided by 0 to 1.
    return corr.mean()

  def getDifferenceAverageFeatureValue(self):
    r"""
    **8. Difference Average**

    .. math::
      \textit{difference average} = \displaystyle\sum^{N_g-1}_{k=0}{kp_{x-y}(k)}

    Difference Average measures the relationship between occurrences of pairs
    with similar intensity values and occurrences of pairs with differing intensity
    values.
    """
    pxSuby = self.coefficients['pxSuby']
    kValuesDiff = self.coefficients['kValuesDiff']
    diffavg = numpy.sum((kValuesDiff[:, None] * pxSuby), 0)
    return diffavg.mean()

  def getDifferenceEntropyFeatureValue(self):
    r"""
    **9. Difference Entropy**

    .. math::
      \textit{difference entropy} = \displaystyle\sum^{N_g-1}_{k=0}{p_{x-y}(k)\log_2\big(p_{x-y}(k)+\epsilon\big)}

    Difference Entropy is a measure of the randomness/variability
    in neighborhood intensity value differences.
    """
    pxSuby = self.coefficients['pxSuby']
    eps = self.coefficients['eps']
    difent = (-1) * numpy.sum((pxSuby * numpy.log2(pxSuby + eps)), 0)
    return difent.mean()

  def getDifferenceVarianceFeatureValue(self):
    r"""
    **10. Difference Variance**

    .. math::
      \textit{difference variance} = \displaystyle\sum^{N_g-1}_{k=0}{(k-DA)^2p_{x-y}(k)}

    Difference Variance is a measure of heterogeneity that places higher weights on
    differing intensity level pairs that deviate more from the mean.
    """
    pxSuby = self.coefficients['pxSuby']
    kValuesDiff = self.coefficients['kValuesDiff']
    diffavg = numpy.sum((kValuesDiff[:, None] * pxSuby), 0, keepdims=True)
    diffvar = numpy.sum((pxSuby * ((kValuesDiff[:, None] - diffavg) ** 2)), 0)
    return diffvar.mean()

  @deprecated
  def getDissimilarityFeatureValue(self):
    r"""
    **DEPRECATED. Dissimilarity**

    .. math::

      \textit{dissimilarity} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{|i-j|p(i,j)}

    .. warning::
      This feature has been deprecated, as it is mathematically equal to Difference Average
      :py:func:`~radiomics.glcm.RadiomicsGLCM.getDifferenceAverageFeatureValue()`.
      See :ref:`here <radiomics-excluded-dissimilarity-label>` for the proof. **Enabling this feature will result in the
      logging of a DeprecationWarning (does not interrupt extraction of other features), no value is calculated for this features**
    """
    raise DeprecationWarning('GLCM - Dissimilarity is mathematically equal to GLCM - Difference Average, '
                             'see http://pyradiomics.readthedocs.io/en/latest/removedfeatures.html for more details')

  def getJointEnergyFeatureValue(self):
    r"""
    **11. Joint Energy**

    .. math::
      \textit{joint energy} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{\big(p(i,j)\big)^2}

    Energy is a measure of homogeneous patterns
    in the image. A greater Energy implies that there are more instances
    of intensity value pairs in the image that neighbor each other at
    higher frequencies.

    .. note::
      Defined by IBSI as Angular Second Moment.
    """
    ene = numpy.sum((self.P_glcm ** 2), (0, 1))
    return ene.mean()

  def getJointEntropyFeatureValue(self):
    r"""
    **12. Joint Entropy**

    .. math::
      \textit{joint entropy} = -\displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}
      {p(i,j)\log_2\big(p(i,j)+\epsilon\big)}


    Joint entropy is a measure of the randomness/variability in neighborhood intensity values.

    .. note::
      Defined by IBSI as Joint entropy
    """
    ent = self.coefficients['HXY']
    return ent.mean()

  @deprecated
  def getHomogeneity1FeatureValue(self):
    r"""
    **DEPRECATED. Homogeneity 1**

    .. math::

      \textit{homogeneity 1} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{\frac{p(i,j)}{1+|i-j|}}

    .. warning::
      This feature has been deprecated, as it is mathematically equal to Inverse Difference
      :py:func:`~radiomics.glcm.RadiomicsGLCM.getIdFeatureValue()`. **Enabling this feature will result in the
      logging of a DeprecationWarning (does not interrupt extraction of other features), no value is calculated for this features**
    """
    raise DeprecationWarning('GLCM - Homogeneity 1 is mathematically equal to GLCM - Inverse Difference, '
                             'see documentation of the GLCM feature class (section "Radiomic Features") for more details')

  @deprecated
  def getHomogeneity2FeatureValue(self):
    r"""
    **DEPRECATED. Homogeneity 2**

    .. math::

      \textit{homogeneity 2} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{\frac{p(i,j)}{1+|i-j|^2}}

    .. warning::
      This feature has been deprecated, as it is mathematically equal to Inverse Difference Moment
      :py:func:`~radiomics.glcm.RadiomicsGLCM.getIdmFeatureValue()`. **Enabling this feature will result in the
      logging of a DeprecationWarning (does not interrupt extraction of other features), no value is calculated for this features**
    """
    raise DeprecationWarning('GLCM - Homogeneity 2 is mathematically equal to GLCM - Inverse Difference Moment, '
                             'see documentation of the GLCM feature class (section "Radiomic Features") for more details')

  def getImc1FeatureValue(self):
    r"""
    **13. Informal Measure of Correlation (IMC) 1**

    .. math::

      \textit{IMC 1} = \frac{HXY-HXY1}{\max\{HX,HY\}}

    .. note::

      In the case where both HX and HY are 0 (as is the case in a flat region), an arbitrary value of 0 is returned to
      prevent a division by 0. This is done on a per-angle basis
    """
    eps = self.coefficients['eps']
    HX = self.coefficients['HX']
    HY = self.coefficients['HY']
    HXY = self.coefficients['HXY']
    HXY1 = self.coefficients['HXY1']

    div = numpy.max(([HX, HY]), 0)

    imc1 = (HXY - HXY1) / (div + eps)
    imc1[div == 0] = 0  # Set elements that would be divided by 0 to 0

    return imc1.mean()

  def getImc2FeatureValue(self):
    r"""
    **14. Informal Measure of Correlation (IMC) 2**

    .. math::

      \textit{IMC 2} = \sqrt{1-e^{-2(HXY2-HXY)}}

    .. note::

      In the case where HXY = HXY2, an arbitrary value of 0 is returned to prevent returning complex numbers. This is
      done on a per-angle basis.
    """
    HXY = self.coefficients['HXY']
    HXY2 = self.coefficients['HXY2']

    imc2 = (1 - numpy.e ** (-2 * (HXY2 - HXY))) ** 0.5
    imc2[HXY2 == HXY] = 0

    return imc2.mean()

  def getIdmFeatureValue(self):
    r"""
    **15. Inverse Difference Moment (IDM)**

    .. math::

      \textit{IDM} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{ \frac{p(i,j)}{1+|i-j|^2} }

    IDM (a.k.a Homogeneity 2) is a measure of the local
    homogeneity of an image. IDM weights are the inverse of the Contrast
    weights (decreasing exponentially from the diagonal i=j in the GLCM).
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    idm = numpy.sum((self.P_glcm / (1 + ((numpy.abs(i - j))[:, :, None] ** 2))), (0, 1))
    return idm.mean()

  def getIdmnFeatureValue(self):
    r"""
    **16. Inverse Difference Moment Normalized (IDMN)**

    .. math::

      \textit{IDMN} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}
      { \frac{p(i,j)}{1+\left(\frac{|i-j|^2}{N_g^2}\right)} }

    IDMN (inverse difference moment normalized)  is a measure of the local
    homogeneity of an image. IDMN weights are the inverse of the Contrast
    weights (decreasing exponentially from the diagonal :math:`i=j` in the GLCM).
    Unlike Homogeneity2, IDMN normalizes the square of the difference between
    neighboring intensity values by dividing over the square of the total
    number of discrete intensity values.
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    Ng = self.coefficients['Ng']
    idmn = numpy.sum((self.P_glcm / (1 + (((numpy.abs(i - j))[:, :, None] ** 2) / (Ng ** 2)))), (0, 1))
    return idmn.mean()

  def getIdFeatureValue(self):
    r"""
    **17. Inverse Difference (ID)**

    .. math::

      \textit{ID} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{ \frac{p(i,j)}{1+|i-j|} }

    ID (a.k.a. Homogeneity 1) is another measure of the local homogeneity of an image.
    With more uniform gray levels, the denominator will remain low, resulting in a higher overall value.
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    invDiff = numpy.sum((self.P_glcm / (1 + (numpy.abs(i - j))[:, :, None])), (0, 1))
    return invDiff.mean()

  def getIdnFeatureValue(self):
    r"""
    **18. Inverse Difference Normalized (IDN)**

    .. math::

      \textit{IDN} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}
      { \frac{p(i,j)}{1+\left(\frac{|i-j|}{N_g}\right)} }

    IDN (inverse difference normalized) is another measure of the local
    homogeneity of an image. Unlike Homogeneity1, IDN normalizes the difference
    between the neighboring intensity values by dividing over the total number
    of discrete intensity values.
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    Ng = self.coefficients['Ng']
    idn = numpy.sum((self.P_glcm / (1 + ((numpy.abs(i - j))[:, :, None] / Ng))), (0, 1))
    return idn.mean()

  def getInverseVarianceFeatureValue(self):
    r"""
    **19. Inverse Variance**

    .. math::

      \textit{inverse variance} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{\frac{p(i,j)}{|i-j|^2}},
      i \neq j
    """
    i = self.coefficients['i']
    j = self.coefficients['j']
    maskDiags = numpy.abs(i - j) > 0
    inv = numpy.sum((self.P_glcm[maskDiags] / ((numpy.abs(i - j))[:, :, None] ** 2)[maskDiags]), 0)
    return inv.mean()

  def getMaximumProbabilityFeatureValue(self):
    r"""
    **20. Maximum Probability**

    .. math::

      \textit{maximum probability} = \max\big(p(i,j)\big)

    Maximum Probability is occurrences of the most predominant pair of
    neighboring intensity values.

    .. note::
      Defined by IBSI as Joint maximum
    """
    maxprob = self.P_glcm.max((0, 1))
    return maxprob.mean()

  def getSumAverageFeatureValue(self):
    r"""
    **21. Sum Average**

    .. math::

      \textit{sum average} = \displaystyle\sum^{2N_g}_{k=2}{p_{x+y}(k)k}

    Sum Average measures the relationship between occurrences of pairs
    with lower intensity values and occurrences of pairs with higher intensity
    values.

    .. warning::
      When GLCM is symmetrical, :math:`\mu_x = \mu_y`, and therefore :math:`\text{Sum Average} = \mu_x + \mu_y =
      2 \mu_x = 2 * Joint Average`. See formulas (4.), (5.) and (6.) defined
      :ref:`here <radiomics-excluded-sumvariance-label>` for the proof that :math:`\text{Sum Average} = \mu_x + \mu_y`.
      In the default parameter files provided in the ``examples/exampleSettings``, this feature has been disabled.
    """
    # warn the user if the GLCM is symmetrical and this feature is calculated (as it is then linearly correlated to Joint Average)
    if self.symmetricalGLCM:
      self.logger.warning('GLCM is symmetrical, therefore Sum Average = 2 * Joint Average, only 1 needs to be calculated')

    pxAddy = self.coefficients['pxAddy']
    kValuesSum = self.coefficients['kValuesSum']
    sumavg = numpy.sum((kValuesSum[:, None] * pxAddy), 0)
    return sumavg.mean()

  @deprecated
  def getSumVarianceFeatureValue(self):
    r"""
    **DEPRECATED. Sum Variance**

    .. math::
      \textit{sum variance} = \displaystyle\sum^{2N_g}_{k=2}{(k-SA)^2p_{x+y}(k)}

    .. warning::
      This feature has been deprecated, as it is mathematically equal to Cluster Tendency
      :py:func:`~radiomics.glcm.RadiomicsGLCM.getClusterTendencyFeatureValue()`.
      See :ref:`here <radiomics-excluded-sumvariance-label>` for the proof. **Enabling this feature will result in the
      logging of a DeprecationWarning (does not interrupt extraction of other features), no value is calculated for this features**
    """
    raise DeprecationWarning('GLCM - Sum Variance is mathematically equal to GLCM - Cluster Tendency, '
                             'see http://pyradiomics.readthedocs.io/en/latest/removedfeatures.html for more details')

  def getSumEntropyFeatureValue(self):
    r"""
    **22. Sum Entropy**

    .. math::

      \textit{sum entropy} = \displaystyle\sum^{2N_g}_{k=2}{p_{x+y}(k)\log_2\big(p_{x+y}(k)+\epsilon\big)}

    Sum Entropy is a sum of neighborhood intensity value differences.
    """
    pxAddy = self.coefficients['pxAddy']
    eps = self.coefficients['eps']
    sumentr = (-1) * numpy.sum((pxAddy * numpy.log2(pxAddy + eps)), 0)
    return sumentr.mean()

  def getSumSquaresFeatureValue(self):
    r"""
    **23. Sum of Squares**

    .. math::

      \textit{sum squares} = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1}{(i-\mu_x)^2p(i,j)}

    Sum of Squares or Variance is a measure in the distribution of neigboring intensity level pairs
    about the mean intensity level in the GLCM.

    .. warning::

      This formula represents the variance of the distribution of :math:`i` and is independent from the distribution
      of :math:`j`. Therefore, only use this formula if the GLCM is symmetrical, where
      :math:`p_x(i) = p_y(j) \text{, where } i = j`

    .. note::
      Defined by IBSI as Joint Variance
    """
    if not self.symmetricalGLCM:
      self.logger.warning('The formula for GLCM - Sum of Squares assumes that the GLCM is symmetrical, but this is not the case.')
    i = self.coefficients['i']
    ux = self.coefficients['ux']
    # Also known as Variance
    ss = numpy.sum((self.P_glcm * ((i[:, :, None] - ux) ** 2)), (0, 1))
    return ss.mean()
