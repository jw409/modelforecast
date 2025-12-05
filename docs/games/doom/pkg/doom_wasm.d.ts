/* tslint:disable */
/* eslint-disable */

export class DoomGame {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a new DOOM game instance
   */
  constructor();
  /**
   * Initialize the renderer with a canvas element
   */
  init_renderer(canvas: HTMLCanvasElement, crt_effect: boolean): void;
  /**
   * Load a WAD file from bytes
   */
  load_wad(data: Uint8Array): void;
  /**
   * Start the game
   */
  start(): void;
  /**
   * Stop the game
   */
  stop(): void;
  /**
   * Run one frame of the game loop
   */
  frame(): void;
  /**
   * Warp to a specific level
   */
  warp(episode: number, map: number): void;
  /**
   * Toggle god mode
   */
  god_mode(): void;
  /**
   * Give all weapons and ammo
   */
  give_all(): void;
  /**
   * Toggle no-clip
   */
  noclip(): void;
}

export function init_panic_hook(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly init_panic_hook: () => void;
  readonly __wbg_doomgame_free: (a: number, b: number) => void;
  readonly doomgame_new: () => number;
  readonly doomgame_init_renderer: (a: number, b: any, c: number) => [number, number];
  readonly doomgame_load_wad: (a: number, b: number, c: number) => [number, number];
  readonly doomgame_start: (a: number) => void;
  readonly doomgame_stop: (a: number) => void;
  readonly doomgame_frame: (a: number) => void;
  readonly doomgame_warp: (a: number, b: number, c: number) => void;
  readonly doomgame_god_mode: (a: number) => void;
  readonly doomgame_give_all: (a: number) => void;
  readonly doomgame_noclip: (a: number) => void;
  readonly wasm_bindgen__convert__closures_____invoke__h51b0e6797fb83016: (a: number, b: number, c: any) => void;
  readonly wasm_bindgen__closure__destroy__h41815e090dd5f943: (a: number, b: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
