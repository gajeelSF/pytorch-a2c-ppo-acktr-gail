import numpy as np

class Walker2dParamManager:
    def __init__(self, env):
        self.env = env
        self.mass_range = [2.0, 7.0]
        self.fric_range = [0.5, 2.0]
        self.restitution_range = [0.5, 1.0]
        self.solimp_range = [0.8, 0.99]
        self.solref_range = [0.001, 0.02]
        self.armature_range = [0.05, 0.98]
        self.tilt_z_range = [-0.18, 0.18]

        self.controllable_param = [1,2,3,4,5,6,7]
        self.activated_param = [1,2,3,4,5,6,7]

        self.param_dim = len(self.activated_param)
        self.sampling_selector = None
        self.selector_target = -1


    def get_params(self):

        mass_param = []
        for bid in range(1, 8):
            cur_mass = self.env.model.body_mass[bid]
            mass_param.append((cur_mass - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0]))

        cur_friction = self.env.model.geom_friction[-1][0]
        friction_param = (cur_friction - self.fric_range[0]) / (self.fric_range[1] - self.fric_range[0])

        cur_restitution = self.env.model.geom_solref[-1][1]
        rest_param = (cur_restitution - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_solimp = self.env.model.geom_solimp[-1][0]
        solimp_param = (cur_solimp - self.solimp_range[0]) / (self.solimp_range[1] - self.solimp_range[0])

        cur_solref = self.env.model.geom_solref[-1][0]
        solref_param = (cur_solref - self.solref_range[0]) / (self.solref_range[1] - self.solref_range[0])

        cur_armature = self.env.model.dof_armature[-1]
        armature_param = (cur_armature - self.armature_range[0]) / (self.armature_range[1] - self.armature_range[0])

        #cur_tiltz = self.env.tilt_z
        #tiltz_param = (cur_tiltz - self.tilt_z_range[0]) / (self.tilt_z_range[1] - self.tilt_z_range[0])

        params = np.array(mass_param + [friction_param, rest_param ,solimp_param, solref_param, armature_param])[self.activated_param]
        return params

    def set_simulator_parameters(self, x):
        cur_id = 0
        for bid in range(0, 7):
            if bid in self.controllable_param:
                mass = x[cur_id] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
                self.env.model.body_mass[bid] = mass
                cur_id += 1

        if 7 in self.controllable_param:
            friction = x[cur_id] * (self.fric_range[1] - self.fric_range[0]) + self.fric_range[0]
            self.env.model.geom_friction[-1][0] = friction
            cur_id += 1
        if 8 in self.controllable_param:
            restitution = x[cur_id] * (self.restitution_range[1] - self.restitution_range[0]) + \
                            self.restitution_range[0]
            for bn in range(len(self.env.model.geom_solref)):
                self.env.model.geom_solref[bn][1] = restitution
            cur_id += 1
        if 9 in self.controllable_param:
            solimp = x[cur_id] * (self.solimp_range[1] - self.solimp_range[0]) + \
                            self.solimp_range[0]
            for bn in range(len(self.env.model.geom_solimp)):
                self.env.model.geom_solimp[bn][0] = solimp
                self.env.model.geom_solimp[bn][1] = solimp
            cur_id += 1
        if 10 in self.controllable_param:
            solref = x[cur_id] * (self.solref_range[1] - self.solref_range[0]) + \
                            self.solref_range[0]
            for bn in range(len(self.env.model.geom_solref)):
                self.env.model.geom_solref[bn][0] = solref
            cur_id += 1
        if 11 in self.controllable_param:
            armature = x[cur_id] * (self.armature_range[1] - self.armature_range[0]) + \
                            self.armature_range[0]
            for dof in range(3, 6):
                self.env.model.dof_armature[dof] = armature
            cur_id += 1
        if 12 in self.controllable_param:
            tiltz = x[cur_id] * (self.tilt_z_range[1] - self.tilt_z_range[0]) + self.tilt_z_range[0]
            self.env.tilt_z = tiltz
            self.env.model.opt.gravity[:] = [9.81 * np.sin(tiltz), 0.0, -9.81 * np.cos(tiltz)]
            cur_id += 1


    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_params()))
        if self.sampling_selector is not None:
            while not self.sampling_selector.classify(np.array([x])) == self.selector_target:
                x = np.random.uniform(0, 1, len(self.get_params()))
        return x
