SCENE=$1
STYLE=$2
echo "Enter the Style image Indexes you want for your respective objects (space separated)" # 14: starry
read -a style_idxs

echo "Enter the COCO object Indexes you want to be styled (space separated)" # 72: TV
read -a class_idxs

data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg

if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff.json
fi

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --style_idxs "${style_idxs[@]}" \
                --class_idxs "${class_idxs[@]}" \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave