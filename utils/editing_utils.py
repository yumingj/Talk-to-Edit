def edit_target_attribute(opt,
                          attribute_dict,
                          edit_labels,
                          round_idx,
                          latent_code,
                          edited_latent_code,
                          field_model,
                          editing_logger=None,
                          print_intermediate_result=False,
                          display_img=False):
    """
    Input: current attribute labels, how to edit
    Output: updated attribute labels
    """

    edit_attr_name = edit_labels['attribute']
    if edit_attr_name is None:
        # dialog_logger.info('No edit in the current round')
        exception_mode = 'normal'
        return attribute_dict, exception_mode, latent_code, edited_latent_code

    # define network
    field_model.target_attr_idx = int(opt['attr_to_idx'][edit_attr_name])
    field_model.load_network(opt['pretrained_field'][edit_attr_name])

    latent_code, edited_latent_code, saved_label, exception_mode = \
        field_model.continuous_editing_with_target(
            latent_codes=latent_code,
            target_cls=edit_labels['target_score'],
            save_dir=opt['path']['visualization'],
            editing_logger=editing_logger,
            edited_latent_code=edited_latent_code,
            prefix=f'edit_order_{str(round_idx)}',
            print_intermediate_result=print_intermediate_result,
            display_img=display_img)

    latent_code = latent_code.cpu().numpy()

    # update attribtue_dict
    for idx, (attr, old_label) in enumerate(list(attribute_dict.items())):
        new_label = int(saved_label[idx])
        if field_model.target_attr_idx != idx and new_label != old_label:
            pass
        attribute_dict[attr] = new_label

    return attribute_dict, exception_mode, latent_code, edited_latent_code
