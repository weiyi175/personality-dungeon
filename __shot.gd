extends Node2D
func _ready() -> void:
	var demo: Node = preload("res://src/ui/SkeletonCutoutDemo.tscn").instantiate()
	add_child(demo)
	await get_tree().process_frame
	var vp := get_viewport_rect().size
	var cw := 380
	var cx := int(vp.x * 0.5) - cw / 2
	var cy := int(vp.y * 0.10)
	var ch := int(vp.y) - cy - 4
	var crop := Rect2i(cx, cy, cw, ch)
	var shots: Array[Image] = []
	for i in 4:
		for j in 14:
			await get_tree().process_frame
		await RenderingServer.frame_post_draw
		shots.append(get_viewport().get_texture().get_image().get_region(crop))
	var comp := Image.create(cw * 4, ch, false, shots[0].get_format())
	for i in shots.size():
		comp.blit_rect(shots[i], Rect2i(0, 0, cw, ch), Vector2i(i * cw, 0))
	comp.save_png("res://_shot_front.png")
	get_tree().quit()
