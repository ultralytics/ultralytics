from ultralytics import YOLO


model = YOLO('runs/detect/train5/weights/best.pt')
# model.val().plot
model.val().box.map

# metrics = model.val() 
# curves = model.val.curves_results
# ap_values = metrics.box.all_ap
# print("ap_values")
# print(ap_values)
# print(len(ap_values))
# metric_keys = metrics.keys
# # key_values = metrics.keys
# print("keys_values")
# precision_value = metrics.results_dict[metric_keys[0]] 
# print(f"Precision: {precision_value}")
# # print(prec)
# # print(len(prec))

# # print(rec)
# # print(len(rec))


# recall_values, precision_values, _ , _ =metrics.curves_results[0]  # Assuming recall is the first curve
# # recall_values = [curve[0] for curve in curves if curve[2] == "Recall"]
# print("recall")
# print(recall_values)
# print(len(recall_values))
# # precision_values = [curve[1] for curve in curves if curve[3] == "Precision"]
# print("precision")
# print(precision_values)
# print(len(precision_values))
# plot_pr_curve(recall_values,precision_values, onplot)

# results = model.predict(source="pe_module_24_1_26/images/test", show=True)

# for idx, frame in enumerate(results):
#     img_path = os.path.join("./pr_images", 'img_'+str(idx)+'.png')
#     cv2.imwrite(img_path, frame.plot(font_size=0.1))