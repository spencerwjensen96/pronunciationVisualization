import { ChangeDetectorRef, Component } from '@angular/core';
import * as Meyda from "meyda";

@Component({
  selector: 'app-audio-capture',
  templateUrl: './audio-capture.component.html',
  styleUrls: ['./audio-capture.component.scss']
})
export class AudioCaptureComponent{

  mediaRecorder!: MediaRecorder;
  audioContext = new AudioContext();
  sourceRef: any;
  audioBlob: Blob;
  audioUrl: string;

  chunks: Blob[] = [];
  isRecording = false;
  mfcc: any[] = new Array(13);
  f1: number = 0;
  f2: number = 0;

  constructor(private cdr: ChangeDetectorRef) { 
    this.audioBlob = new Blob([], { type: 'audio/wav' });
    this.audioUrl = URL.createObjectURL(this.audioBlob);
  }

  updatePointPosition(x: number, y: number) {
    this.f1 = x * 5;
    this.f2 = y * 5;
  }

  updateMFCC(array: any[]) {
    this.mfcc = array;
    this.updatePointPosition(array[1], array[2]);
    this.cdr.detectChanges();
  }

  async startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);
      this.chunks = [];

      this.mediaRecorder.addEventListener('dataavailable', (event) => {
        this.chunks.push(event.data);
      });

      this.mediaRecorder.addEventListener('stop', () => {
        this.isRecording = false;
        this.sourceRef.disconnect();
      });

      this.mediaRecorder.start();
      this.isRecording = true;
      const source = this.audioContext.createMediaStreamSource(this.mediaRecorder.stream);
      this.sourceRef = source;
      //this.sourceRef.connect(this.audioContext.destination);

      const analyzer = Meyda.default.createMeydaAnalyzer({
        audioContext: this.audioContext,
        source: this.sourceRef,
        bufferSize: 512,
        featureExtractors: ["mfcc"],
        callback: (features: any) => {
          this.updateMFCC(features.mfcc);
          console.log(features.mfcc);
        },
      });
      analyzer.start();
      
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
    }
  }
}