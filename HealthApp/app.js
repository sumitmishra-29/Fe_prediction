import React, { useState, useRef } from 'react';
import { View, Text, TextInput, Button, StyleSheet } from 'react-native';
import { RNCamera } from 'react-native-camera';
import axios from 'axios';

const App = () => {
    const [name, setName] = useState('');
    const [sex, setSex] = useState('');
    const [age, setAge] = useState('');
    const [bloodGroup, setBloodGroup] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const cameraRef = useRef(null);

    const handleStartRecording = async () => {
        if (cameraRef.current) {
            setIsRecording(true);
            const options = { quality: RNCamera.Constants.VideoQuality['480p'], maxDuration: 20 };
            const data = await cameraRef.current.recordAsync(options);
            console.log(data.uri);
            handleSubmit(data.uri);
        }
    };

    const handleStopRecording = () => {
        if (cameraRef.current && isRecording) {
            cameraRef.current.stopRecording();
            setIsRecording(false);
        }
    };

    const handleSubmit = async (videoUri) => {
        const formData = new FormData();
        formData.append('name', name);
        formData.append('sex', sex);
        formData.append('age', age);
        formData.append('bloodGroup', bloodGroup);
        formData.append('video', {
            uri: videoUri,
            type: 'video/mp4',
            name: 'recording.mp4'
        });

        try {
            await axios.post('http://49.43.162.16/32:3000/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            alert('Data uploaded successfully');
        } catch (error) {
            console.error(error);
            alert('Failed to upload data');
        }
    };

    return (
        <View style={styles.container}>
            <TextInput placeholder="Name" value={name} onChangeText={setName} style={styles.input} />
            <TextInput placeholder="Sex" value={sex} onChangeText={setSex} style={styles.input} />
            <TextInput placeholder="Age" value={age} onChangeText={setAge} style={styles.input} keyboardType="numeric" />
            <TextInput placeholder="Blood Group" value={bloodGroup} onChangeText={setBloodGroup} style={styles.input} />

            {!isRecording ? (
                <Button title="Next" onPress={() => setIsRecording(true)} />
            ) : (
                <View style={styles.cameraContainer}>
                    <RNCamera ref={cameraRef} style={styles.preview} type={RNCamera.Constants.Type.back} />
                    <Button title="Start Recording" onPress={handleStartRecording} />
                    <Button title="Stop Recording" onPress={handleStopRecording} />
                    <Button title="Submit" onPress={handleStopRecording} />
                </View>
            )}
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        padding: 16,
    },
    input: {
        height: 40,
        borderColor: 'gray',
        borderWidth: 1,
        marginBottom: 12,
        padding: 8,
    },
    cameraContainer: {
        flex: 1,
        justifyContent: 'center',
    },
    preview: {
        flex: 1,
        justifyContent: 'flex-end',
        alignItems: 'center',
    },
});

export default App;
