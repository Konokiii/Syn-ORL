MUJOCO_FULL_DESCRIPTION = {
    'nickname': 'mjc_full',
    'state': {
        'walker2d': ['z-coordinate (height) of the torso', 'angle of the torso', 'angle of the right thigh joint',
                     'angle of the right leg joint', 'angle of the right foot joint', 'angle of the left thigh joint',
                     'angle of the left leg joint', 'angle of the left foot joint',
                     'velocity of the x-coordinate of the torso',
                     'velocity of the z-coordinate (height) of the torso', 'angular velocity of the angle of the torso',
                     'angular velocity of the right thigh hinge', 'angular velocity of the right leg hinge',
                     'angular velocity of the right foot hinge', 'angular velocity of the left thigh hinge',
                     'angular velocity of the left leg hinge', 'angular velocity of the left foot hinge'],
        'halfcheetah': ['z-coordinate of the front tip', 'angle of the front tip', 'angle of the back thigh',
                        'angle of the back shin', 'angle of the back foot', 'angle of the front thigh',
                        'angle of the front shin', 'angle of the front foot',
                        'velocity of the x-coordinate of front tip',
                        'velocity of the z-coordinate of front tip', 'angular velocity of the front tip',
                        'angular velocity of the back thigh', 'angular velocity of the back shin',
                        'angular velocity of the back foot', 'angular velocity of the front thigh',
                        'angular velocity of the front shin', 'angular velocity of the front foot']
    },
    'action': {
        'walker2d': ['Torque applied on the right thigh rotor', 'Torque applied on the right leg rotor',
                     'Torque applied on the right foot rotor', 'Torque applied on the left thigh rotor',
                     'Torque applied on the left leg rotor', 'Torque applied on the left foot rotor'],
        'halfcheetah': ['Torque applied on the back thigh rotor', 'Torque applied on the back shin rotor',
                        'Torque applied on the back foot rotor', 'Torque applied on the front thigh rotor',
                        'Torque applied on the front shin rotor', 'Torque applied on the front foot rotor'],
    }
}

MUJOCO_SHORT_DESCRIPTION = {
    'nickname': 'mjc_short',
    'state': {
        'walker2d': ['height of body', 'angle of body', 'angle of right thigh',
                     'angle of right leg', 'angle of right foot', 'angle of left thigh',
                     'angle of left leg', 'angle of left foot', 'horizontal velocity of body',
                     'vertical velocity of body', 'angular velocity of body',
                     'angular velocity of right thigh', 'angular velocity of right leg',
                     'angular velocity of right foot', 'angular velocity of left thigh',
                     'angular velocity of left leg', 'angular velocity of left foot'],
        'halfcheetah': ['height of front tip', 'angle of front tip', 'angle of back thigh',
                        'angle of back shin', 'angle of back foot', 'angle of front thigh',
                        'angle of front shin', 'angle of front foot',
                        'horizontal velocity of front tip',
                        'vertical velocity of front tip', 'angular velocity of front tip',
                        'angular velocity of back thigh', 'angular velocity of back shin',
                        'angular velocity of back foot', 'angular velocity of front thigh',
                        'angular velocity of front shin', 'angular velocity of front foot']
    },
    'action': {
        'walker2d': ['torque of right thigh', 'torque of right leg',
                     'torque of right foot', 'torque of left thigh',
                     'torque of left leg', 'torque of left foot'],
        'halfcheetah': ['torque of back thigh', 'torque of back shin',
                        'torque of back foot', 'torque of front thigh',
                        'torque of front shin', 'torque of front foot'],
    }
}

MUJOCO_UNIT = {
    'nickname': 'mjc_unit',
    'state': {
        'walker2d': ['m'] + ['rad' for _ in range(7)] + ['m/s', 'm/s'] + ['rad/s' for _ in range(7)],
        'halfcheetah': ['m'] + ['rad' for _ in range(7)] + ['m/s', 'm/s'] + ['rad/s' for _ in range(7)]
    },
    'action': {
        'walker2d': ['N m' for _ in range(6)],
        'halfcheetah': ['N m' for _ in range(6)]
    }
}
