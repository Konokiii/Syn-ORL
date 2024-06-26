import os
import pickle
from transformers import AutoModel

for lm in ['sentence-transformers/all-mpnet-base-v2']:
    language_model = AutoModel.from_pretrained(lm)
    for env in ['hopper', 'halfcheetah', 'walker2d']:
        for data in ['medium-replay', 'medium', 'medium-expert']:
            for prefix in ['mjc_re']:
                for suffix in ['mjc_unit']:
                    file_prefix = '%s_%s_%s_%s_%s' % (
                        env, data,
                        prefix, suffix,
                        language_model.__class__.__name__
                    )
                    state_file = './' + file_prefix + '_S.pkl'
                    action_file = './' + file_prefix + '_A.pkl'
                    new_file = './' + file_prefix + '.pkl'
                    data_dict = {}
                    try:
                        with open(state_file, 'rb') as file:
                            states, next_states = pickle.load(file)
                            print(f'Get state embeddings from {state_file}')
                        with open(action_file, 'rb') as file:
                            actions = pickle.load(file)
                            print(f'Get action embeddings from {action_file}')
                        data_dict = {'observations': states, 'next_observations': next_states, 'actions': actions}
                        with open(new_file, 'wb') as file:
                            pickle.dump(data_dict, file)
                            print(f'Writing to {new_file}')
                    except Exception as e:
                        print(e)