import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AudioCaptureComponent } from './audio-capture.component';

describe('AudioCaptureComponent', () => {
  let component: AudioCaptureComponent;
  let fixture: ComponentFixture<AudioCaptureComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ AudioCaptureComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(AudioCaptureComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
