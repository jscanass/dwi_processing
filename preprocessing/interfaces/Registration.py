from nipype.interfaces.base import (BaseInterfaceInputSpec, File, 
                                    TraitedSpec, BaseInterface)

class RegistrationInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='The input 4D diffusion-weighted image file')
    static = File(
        exists=True,
        mandatory=True,
        desc='T1 or B0 static reference file')
    out_file = File(
        mandatory=True,
        desc='output file')

class RegistrationOutputSpec(TraitedSpec):
    out_file = File(genfile=True) #what is genfile?

class Registration(BaseInterface):
    input_spec = RegistrationInputSpec
    output_spec = RegistrationOutputSpec

    def _run_interface(self, runtime):

        registration_proxy(
            self.inputs.in_file,
            self.inputs.static,
            self.inputs.out_file
        )

        return runtime
    
    def _list_outputs(self):
        return {'out_file': self.inputs.out_file}    

def registration_proxy(in_file,static,out_file): 
    
    """
    http://nipy.org/dipy/examples_built/affine_registration_3d.html
    in_file --> moving
    
    static and moving = path 
    
    """
    import time
    import numpy as np
    import nibabel as nb
    
    import matplotlib.pyplot as plt
    from dipy.viz import regtools 
    
    from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
    from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
    
    
    
    t0_time = time.time()

    print('---> I. Translation of the moving image towards the static image')

    #condition if we have a path or a nifti file

    static_img = nb.load(static)
    static = static_img.get_data()
    static_grid2world = static_img.affine

    moving_img = nb.load(in_file)
    moving = np.array(moving_img.get_data())[..., 0]
    moving_grid2world = moving_img.affine

    # resample for have the same number of voxels
    
    print('---> Resembling the moving image on a grid of the same dimensions as the static image')
    
    identity = np.eye(4)
    affine_map = AffineMap(identity,
                       static.shape, static_grid2world,
                       moving.shape, moving_grid2world)
    resampled = affine_map.transform(moving)
    regtools.overlay_slices(static, resampled, None, 0,
                        "Static", "Moving", "resampled_0.png")


    regtools.overlay_slices(static, resampled, None, 1,
                        "Static", "Moving", "resampled_1.png")


    regtools.overlay_slices(static, resampled, None, 2,
                        "Static", "Moving", "resampled_2.png")
    plt.show()

    #centers of mass registration
    
    print('---> Aligning the centers of mass of the two images')
    
    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                      moving, moving_grid2world)
    transformed = c_of_mass.transform(moving)
    
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_com_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_com_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_com_2.png")
    plt.show()

    
    print('---> II. Refine  by looking for an affine transform')
    
    #affine transform
    #parameters??    
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)


    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    
    print('---> Computing Affine Registration (non-convex optimization)')
    
    affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)
    
    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)
    
    transformed = translation.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_trans_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_trans_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_trans_2.png")  
    plt.show()

    
    print('--->III. Refining with a rigid transform')
    
    #rigid transform 
    
    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)
    
    
    transformed = rigid.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                            "Static", "Transformed", "transformed_rigid_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                            "Static", "Transformed", "transformed_rigid_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                            "Static", "Transformed", "transformed_rigid_2.png")
    plt.show()

    
    print('--->IV. Refining with a full afine transform (translation, rotation, scale and shear)')
    
    #full affine transform 
    
    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=starting_affine)
    
    transformed = affine.transform(moving)
    regtools.overlay_slices(static, transformed, None, 0,
                        "Static", "Transformed", "transformed_affine_0.png")
    regtools.overlay_slices(static, transformed, None, 1,
                        "Static", "Transformed", "transformed_affine_1.png")
    regtools.overlay_slices(static, transformed, None, 2,
                        "Static", "Transformed", "transformed_affine_2.png")
    plt.show()

    
    # Save the new data in a new NIfTI image
    nb.Nifti1Image(transformed, static_img.affine).to_filename(out_file)
    
    #name = os.path.splitext(basename(moving_path))[0] + '_affine_reg'
    #nib.save(nib.Nifti1Image(transformed, np.eye(4)), name)
    t1_time = time.time()
    total_time = t1_time-t0_time
    print('Total time:' + str(total_time))
    print('Translated file now is here: %s' % out_file)
    return print('Successfully affine registration applied')