import { Component, OnInit, ViewChild, ElementRef, ChangeDetectorRef } from '@angular/core';
import * as Meyda from "meyda";

@Component({
  selector: 'app-audio-player',
  templateUrl: './audio-player.component.html',
  styleUrls: ['./audio-player.component.scss']
})
export class AudioPlayerComponent implements OnInit{
  @ViewChild('audioElement') audioElement!: ElementRef<HTMLMediaElement>;

  audioFileUrl = '/assets/audio/take_a_picture.wav';
  audioContext = new AudioContext();
  mfcc: any[] = new Array(13);
  f1: number = 0;
  f2: number = 0;

  constructor(private cdr: ChangeDetectorRef) { }

  // Method to update the position of the plotted point
  updatePointPosition(x: number, y: number) {
    this.f1 = x * 5;
    this.f2 = y * 5;
  }

  updateMFCC(array: any[]) {
    this.mfcc = array;
    this.updatePointPosition(array[1], array[2]);
    this.cdr.detectChanges();
  }

  ngOnInit(): void {
    // const htmlAudioElement = document.getElementById("audio") as HTMLMediaElement;
    
  }
  ngAfterViewInit(): void {
    const source = this.audioContext.createMediaElementSource(this.audioElement.nativeElement);
    source.connect(this.audioContext.destination);

    if (typeof Meyda === "undefined") {
      console.log("Meyda could not be found! Have you included it?");
    } else {
      const analyzer = Meyda.default.createMeydaAnalyzer({
        audioContext: this.audioContext,
        source: source,
        bufferSize: 512,
        featureExtractors: ["mfcc"],
        callback: (features: any) => {
          this.updateMFCC(features.mfcc);
          //console.log(features.mfcc);
        },
      });
      analyzer.start();
    }
  }
}
