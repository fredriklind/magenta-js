// tslint:disable:no-any
import * as mm from '../src/index';

declare global {

    interface Window { mm: any; }
}

window.mm = mm;

import * as tf from '@tensorflow/tfjs';

import {logging, MidiMe, MusicVAE, NoteSequence} from '../src/index';
import {quantizeNoteSequence} from '../src/core/sequences';

const melModel = new mm.MidiMe({epochs: 100});
melModel.initialize();
const trioModel = new mm.MidiMe({epochs: 300});
trioModel.initialize();

async function train(
    mel: NoteSequence,
    vae: MusicVAE, midime: MidiMe): Promise<mm.INoteSequence[]> {

    // 1. Encode the input into MusicVAE, get back a z.
    const quantizedMels: NoteSequence[] = [];
    const mels = [mel];
    mels.forEach((m) => quantizedMels.push(quantizeNoteSequence(m, 4)));

    // 1b. Split this sequence into 32 bar chunks.
    let chunks: NoteSequence[] = [];
    quantizedMels.forEach((m) => {
        const length = 16 * 2; // 2 bars
        const melChunks = mm.sequences.split(mm.sequences.clone(m), length);
        chunks = chunks.concat(melChunks);
    });
    const z = await vae.encode(chunks);  // shape of z is [chunks, 256]

    // 2. Use that z as input to train MidiMe.
    // Reconstruction before training.
    const z1 = midime.predict(z) as tf.Tensor2D;
    /*const ns1 = */ await vae.decode(z1);
    z1.dispose();

    // 3. Train!
    // tslint:disable-next-line:no-any
    await midime.train(z, async (epoch: number, logs: any) => {
        logging.log('Training!', logs);
    });

    const z2 = await midime.sample() as tf.Tensor2D;
    // z2.dispose();
    return await vae.decode(z2);
}

async function sample(
    vae: MusicVAE, midime: MidiMe): Promise<mm.INoteSequence[]> {
    const z2 = await midime.sample() as tf.Tensor2D;
    // z2.dispose();
    return await vae.decode(z2);
}

window.mm.trainAndSampleMidiMe = train;
window.mm.sampleMidiMe = sample;
