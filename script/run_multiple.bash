proj_dir="/home/justin/code/LieNeurons/"
config_path="config/"
num_experiment=4
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
NC='\033[0m' # No Color
export proj_dir

# ============================================================================================================================================================================================
# invariant tasks
# mlp
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running invariant task: mlp $iter %%--------------------"
#   export iter
#   yq e -i '.train_data_path = "/home/justin/code/LieNeurons/data/sl3_inv_data/sl3_inv_10000_s_05_train_data.npz"' $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_inv_mlp_"+env(iter)'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_inv_mlp_"+env(iter)'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.model_type = "MLP"'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.initial_learning_rate = 0.0001'  $config_path"sl3_inv/training_param.yaml"
#   python experiment/sl3_inv_train.py
# done
# # invariant tasks
# # LN-LR
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running invariant task: LN-LR $iter %%--------------------"
#   export iter
#   yq e -i '.train_data_path = "/home/justin/code/LieNeurons/data/sl3_inv_data/sl3_inv_10000_s_05_train_data.npz"' $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_inv_relu_"+env(iter)'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_inv_relu_"+env(iter)'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.model_type = "LN_relu"'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.initial_learning_rate = 0.0001'  $config_path"sl3_inv/training_param.yaml"
#   python experiment/sl3_inv_train.py
# done
# # invariant tasks
# # LN-LB
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running invariant task: LN-LB $iter %%--------------------"
#   export iter
#   yq e -i '.train_data_path = "/home/justin/code/LieNeurons/data/sl3_inv_data/sl3_inv_10000_s_05_train_data.npz"' $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_inv_bracket_"+env(iter)'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_inv_bracket_"+env(iter)'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.model_type = "LN_bracket"'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.initial_learning_rate = 0.0001'  $config_path"sl3_inv/training_param.yaml"
#   python experiment/sl3_inv_train.py
# done

# # invariant tasks
# # LN-LR+LB
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running invariant task: LN-LR+LB $iter %%--------------------"
#   export iter
#   yq e -i '.train_data_path = "/home/justin/code/LieNeurons/data/sl3_inv_data/sl3_inv_10000_s_05_train_data.npz"' $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_inv_rb_"+env(iter)'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_inv_rb_"+env(iter)'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.model_type = "LN_relu_bracket"'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.initial_learning_rate = 0.0001'  $config_path"sl3_inv/training_param.yaml"
#   python experiment/sl3_inv_train.py
# done

# # mlp augmented
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running invariant task: mlp augmented $iter %%--------------------"
#   export iter
#   # train_data_path: "/home/justin/code/LieNeurons/data/sl3_inv_data/sl3_inv_10000_s_05_augmented_train_data.npz"
# # test_data_path: "/home/justin/code/LieNeurons/data/sl3_inv_data/sl3_inv_10000_s_05_test_data.npz"
#   yq e -i '.train_data_path = "/home/justin/code/LieNeurons/data/sl3_inv_data/sl3_inv_10000_s_05_augmented_train_data.npz"' $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_inv_mlp_augmented_"+env(iter)'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_inv_mlp_augmented_"+env(iter)'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.model_type = "MLP"'  $config_path"sl3_inv/training_param.yaml"
#   yq e -i '.initial_learning_rate = 0.0001'  $config_path"sl3_inv/training_param.yaml"
#   python experiment/sl3_inv_train.py
# done

# # ============================================================================================================================================================================================
# # equivariant task
# # mlp
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running equivariant task: mlp $iter %%--------------------"
#   export iter
#   yq e -i '.train_data_path = "/home/justin/code/LieNeurons/data/sl3_equiv_lie_bracket_data/sl3_equiv_10000_lie_bracket_2inputs_train_data.npz"' $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_equiv_mlp_"+env(iter)'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_equiv_mlp_"+env(iter)'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.model_type = "MLP"'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.initial_learning_rate = 0.00001'  $config_path"sl3_equiv/training_param.yaml"
#   python experiment/sl3_equiv_train.py
# done

# # equivariant task
# # LN-LR
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running equivariant task: LN-LR $iter %%--------------------"
#   export iter
#   yq e -i '.train_data_path = "/home/justin/code/LieNeurons/data/sl3_equiv_lie_bracket_data/sl3_equiv_10000_lie_bracket_2inputs_train_data.npz"' $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_equiv_relu_"+env(iter)'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_equiv_relu_"+env(iter)'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.model_type = "LN_relu"'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.initial_learning_rate = 0.000001'  $config_path"sl3_equiv/training_param.yaml"
#   python experiment/sl3_equiv_train.py
# done

# # equivariant task
# # LN-LB
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running equivariant task: LN-LB $iter %%--------------------"
#   export iter
#   yq e -i '.train_data_path = "/home/justin/code/LieNeurons/data/sl3_equiv_lie_bracket_data/sl3_equiv_10000_lie_bracket_2inputs_train_data.npz"' $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_equiv_bracket_"+env(iter)'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_equiv_bracket_"+env(iter)'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.model_type = "LN_bracket"'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.initial_learning_rate = 0.00001'  $config_path"sl3_equiv/training_param.yaml"
#   python experiment/sl3_equiv_train.py
# done

# # equivariant task
# # LN-LR+LB
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running equivariant task: LN-LR+LB $iter %%--------------------"
#   export iter
#   yq e -i '.train_data_path = "/home/justin/code/LieNeurons/data/sl3_equiv_lie_bracket_data/sl3_equiv_10000_lie_bracket_2inputs_train_data.npz"' $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_equiv_rb_"+env(iter)'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_equiv_rb_"+env(iter)'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.model_type = "LN_relu_bracket"'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.initial_learning_rate = 0.00001'  $config_path"sl3_equiv/training_param.yaml"
#   python experiment/sl3_equiv_train.py
# done

# equivariant task
# mlp augmented
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running equivariant task: mlp augmented $iter %%--------------------"
#   export iter
#   yq e -i '.train_data_path =  "/home/justin/code/LieNeurons/data/sl3_equiv_lie_bracket_data/sl3_equiv_10000_lie_bracket_2inputs_augmented_train_data.npz"' $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_equiv_mlp_augmented_"+env(iter)'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_equiv_mlp_augmented_"+env(iter)'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.model_type = "MLP"'  $config_path"sl3_equiv/training_param.yaml"
#   yq e -i '.initial_learning_rate = 0.00001'  $config_path"sl3_equiv/training_param.yaml"
#   python experiment/sl3_equiv_train.py
# done

# ============================================================================================================================================================================================
# classification task

# mlp aug
for iter in $(seq 1 $num_experiment); do
  echo -e "--------------------%% Running classification task: mlp augmentation $iter %%--------------------"
  export iter
  yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_cls_mlp_aug_"+env(iter)'  $config_path"platonic_solid_cls/training_param.yaml"
  yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_cls_mlp_aug_"+env(iter)'  $config_path"platonic_solid_cls/training_param.yaml"
  yq e -i '.model_type = "MLP"'  $config_path"platonic_solid_cls/training_param.yaml"
  yq e -i '.initial_learning_rate = 0.0001'  $config_path"platonic_solid_cls/training_param.yaml"
  yq e -i '.train_augmentation = True'  $config_path"platonic_solid_cls/training_param.yaml"
  python experiment/platonic_solid_cls_train.py
done


# mlp
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running classification task: mlp $iter %%--------------------"
#   export iter
#   yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_cls_mlp_"+env(iter)'  $config_path"platonic_solid_cls/training_param.yaml"
#   yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_cls_mlp_"+env(iter)'  $config_path"platonic_solid_cls/training_param.yaml"
#   yq e -i '.model_type = "MLP"'  $config_path"platonic_solid_cls/training_param.yaml"
#   yq e -i '.initial_learning_rate = 0.00001'  $config_path"platonic_solid_cls/training_param.yaml"
#   python experiment/platonic_solid_cls_train.py
# done

# # classification task
# # LN-LR
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running classification task: LN-LR $iter %%--------------------"
#   export iter
#   yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_cls_relu_"+env(iter)'  $config_path"platonic_solid_cls/training_param.yaml"
#   yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_cls_relu_"+env(iter)'  $config_path"platonic_solid_cls/training_param.yaml"
#   yq e -i '.model_type = "LN_relu"'  $config_path"platonic_solid_cls/training_param.yaml"
#   yq e -i '.initial_learning_rate = 0.00001'  $config_path"platonic_solid_cls/training_param.yaml"
#   python experiment/platonic_solid_cls_train.py
# done

# # classification task
# # LN-LB
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running classification task: LN-LB $iter %%--------------------"
#   export iter
#   yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_cls_bracket_"+env(iter)'  $config_path"platonic_solid_cls/training_param.yaml"
#   yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_cls_bracket_"+env(iter)'  $config_path"platonic_solid_cls/training_param.yaml"
#   yq e -i '.model_type = "LN_bracket"'  $config_path"platonic_solid_cls/training_param.yaml"
#   yq e -i '.initial_learning_rate = 0.00001'  $config_path"platonic_solid_cls/training_param.yaml"
#   python experiment/platonic_solid_cls_train.py
# done
# # classification task
# # LN-LR+LB
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running classification task: LN-LR+LB $iter %%--------------------"
#   export iter
#   yq e -i '.model_save_path = strenv(proj_dir)+"weights/rebuttal_cls_rb_"+env(iter)'  $config_path"platonic_solid_cls/training_param.yaml"
#   yq e -i '.log_writer_path = strenv(proj_dir)+"logs/rebuttal_cls_rb_"+env(iter)'  $config_path"platonic_solid_cls/training_param.yaml"
#   yq e -i '.model_type = "LN_relu_bracket"'  $config_path"platonic_solid_cls/training_param.yaml"
#   yq e -i '.initial_learning_rate = 0.00001'  $config_path"platonic_solid_cls/training_param.yaml"
#   python experiment/platonic_solid_cls_train.py
# done

# # ******************************************************************************
# # *******************************Evaluations ***********************************
# # ******************************************************************************
# # invariant tasks
# # mlp
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running invariant task: mlp $iter %%--------------------"
#   export iter
#   yq e -i '.model_path = strenv(proj_dir)+"weights/rebuttal_inv_mlp_"+env(iter)+"_best_test_acc.pt"'  $config_path"sl3_inv/testing_param.yaml"
#   yq e -i '.model_type = "MLP"'  $config_path"sl3_inv/testing_param.yaml"
#   python experiment/sl3_inv_test.py
# done
# # invariant tasks
# # LN-LR
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running invariant task: LN-LR $iter %%--------------------"
#   export iter
#   yq e -i '.model_path = strenv(proj_dir)+"weights/rebuttal_inv_relu_"+env(iter)+"_best_test_acc.pt"'  $config_path"sl3_inv/testing_param.yaml"
#   yq e -i '.model_type = "LN_relu"'  $config_path"sl3_inv/testing_param.yaml"
#   python experiment/sl3_inv_test.py
# done
# # invariant tasks
# # LN-LB
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running invariant task: LN-LB $iter %%--------------------"
#   export iter
#   yq e -i '.model_path = strenv(proj_dir)+"weights/rebuttal_inv_bracket_"+env(iter)+"_best_test_acc.pt"'  $config_path"sl3_inv/testing_param.yaml"
#   yq e -i '.model_type = "LN_bracket"'  $config_path"sl3_inv/testing_param.yaml"
#   python experiment/sl3_inv_test.py
# done

# # invariant tasks
# # LN-LR+LB
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running invariant task: LN-LR+LB $iter %%--------------------"
#   export iter
#   yq e -i '.model_path = strenv(proj_dir)+"weights/rebuttal_inv_rb_"+env(iter)+"_best_test_acc.pt"'  $config_path"sl3_inv/testing_param.yaml"
#   yq e -i '.model_type = "LN_relu_bracket"'  $config_path"sl3_inv/testing_param.yaml"
#   python experiment/sl3_inv_test.py
# done

# # mlp augmented
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running invariant task: mlp augmented $iter %%--------------------"
#   export iter
#   yq e -i '.model_path = strenv(proj_dir)+"weights/rebuttal_inv_mlp_augmented_"+env(iter)+"_best_test_acc.pt"'  $config_path"sl3_inv/testing_param.yaml"
#   yq e -i '.model_type = "MLP"'  $config_path"sl3_inv/testing_param.yaml"
#   python experiment/sl3_inv_test.py
# done

# # ============================================================================================================================================================================================
# # equivariant task
# # mlp
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running equivariant task: mlp $iter %%--------------------"
#   export iter
#   yq e -i '.model_path = strenv(proj_dir)+"weights/rebuttal_equiv_mlp_"+env(iter)+"_best_test_acc.pt"'  $config_path"sl3_equiv/testing_param.yaml"
#   yq e -i '.model_type = "MLP"'  $config_path"sl3_equiv/testing_param.yaml"
#   python experiment/sl3_equiv_test.py
# done

# # equivariant task
# # LN-LR
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running equivariant task: LN-LR $iter %%--------------------"
#   export iter
#   yq e -i '.model_path = strenv(proj_dir)+"weights/rebuttal_equiv_relu_"+env(iter)+"_best_test_acc.pt"'  $config_path"sl3_equiv/testing_param.yaml"
#   yq e -i '.model_type = "LN_relu"'  $config_path"sl3_equiv/testing_param.yaml"
#   python experiment/sl3_equiv_test.py
# done

# # equivariant task
# # LN-LB
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running equivariant task: LN-LB $iter %%--------------------"
#   export iter
#   yq e -i '.model_path = strenv(proj_dir)+"weights/rebuttal_equiv_bracket_"+env(iter)+"_best_test_acc.pt"'  $config_path"sl3_equiv/testing_param.yaml"
#   yq e -i '.model_type = "LN_bracket"'  $config_path"sl3_equiv/testing_param.yaml"
#   python experiment/sl3_equiv_test.py
# done

# # equivariant task
# # LN-LR+LB
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running equivariant task: LN-LR+LB $iter %%--------------------"
#   export iter
#   yq e -i '.model_path = strenv(proj_dir)+"weights/rebuttal_equiv_rb_"+env(iter)+"_best_test_acc.pt"'  $config_path"sl3_equiv/testing_param.yaml"
#   yq e -i '.model_type = "LN_relu_bracket"'  $config_path"sl3_equiv/testing_param.yaml"
#   python experiment/sl3_equiv_test.py
# done

# # equivariant task
# # mlp augmented
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running equivariant task: mlp augmented $iter %%--------------------"
#   export iter
#   yq e -i '.model_path = strenv(proj_dir)+"weights/rebuttal_equiv_mlp_augmented_"+env(iter)+"_best_test_acc.pt"'  $config_path"sl3_equiv/testing_param.yaml"
#   yq e -i '.model_type = "MLP"'  $config_path"sl3_equiv/testing_param.yaml"
#   python experiment/sl3_equiv_test.py
# done

# # ============================================================================================================================================================================================
# # classification task
# # mlp
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running classification task: mlp $iter %%--------------------"
#   export iter
#   yq e -i '.model_path = strenv(proj_dir)+"weights/rebuttal_cls_mlp_"+env(iter)+"_best_test_acc.pt"'  $config_path"platonic_solid_cls/testing_param.yaml"
#   yq e -i '.model_type = "MLP"'  $config_path"platonic_solid_cls/testing_param.yaml"
#   python experiment/platonic_solid_cls_test.py
# done

# # classification task
# # LN-LR
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running classification task: LN-LR $iter %%--------------------"
#   export iter
#   yq e -i '.model_path = strenv(proj_dir)+"weights/rebuttal_cls_relu_"+env(iter)+"_best_test_acc.pt"'  $config_path"platonic_solid_cls/testing_param.yaml"
#   yq e -i '.model_type = "LN_relu"'  $config_path"platonic_solid_cls/testing_param.yaml"
#   python experiment/platonic_solid_cls_test.py
# done

# # classification task
# # LN-LB
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running classification task: LN-LB $iter %%--------------------"
#   export iter
#   yq e -i '.model_path = strenv(proj_dir)+"weights/rebuttal_cls_bracket_"+env(iter)+"_best_test_acc.pt"'  $config_path"platonic_solid_cls/testing_param.yaml"
#   yq e -i '.model_type = "LN_bracket"'  $config_path"platonic_solid_cls/testing_param.yaml"
#   python experiment/platonic_solid_cls_test.py
# done
# # classification task
# # LN-LR+LB
# for iter in $(seq 1 $num_experiment); do
#   echo -e "--------------------%% Running classification task: LN-LR+LB $iter %%--------------------"
#   export iter
#   yq e -i '.model_path = strenv(proj_dir)+"weights/rebuttal_cls_rb_"+env(iter)+"_best_test_acc.pt"'  $config_path"platonic_solid_cls/testing_param.yaml"
#   yq e -i '.model_type = "LN_relu_bracket"'  $config_path"platonic_solid_cls/testing_param.yaml"
#   python experiment/platonic_solid_cls_test.py
# done