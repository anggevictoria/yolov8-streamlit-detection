detected_objects_list = [] # Initialize a list to keep the order of detections


            try:
                for box in boxes:
                                # Extract relevant information from the detection box
                                x1, y1, x2, y2, confidence, class_id = box.data[0][:6]  # Adjust indices if `box.data` has a different structure
                                object_name = model.names[int(class_id)]  # Map class ID to name

                                if object_name not in detected_objects_list:
                                    #add the object to the set and process
                                    detected_objects_list.append(object_name) # Add the object to the list at the last position

                                else:
                                    continue

                                # Use the first item in the list (the first detected object)
                                first_detected_object = detected_objects_list[0]
                                description = helper.generate_description(first_detected_object)
                               
                                # Empty container to hold the message
                                msg_container = st.empty()    
        
                                msg_container.write(f"{first_detected_object} detected: {description}. Ask Streaming chatbot if you want to know more about {detected_objects_list[0]}")
                                time.sleep(3)  # Wait for 3 seconds before clearing the message
                                msg_container.empty()  # Remove the message after 3 seconds

                                # Remove the first item from the set and list after processing
                                detected_objects_list.pop(0)  # Remove the first item in the list
