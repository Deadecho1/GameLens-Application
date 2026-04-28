export {};

declare global {
  interface Window {
    gamelens?: {
      request(method: string, params?: Record<string, unknown>): Promise<any>;
      chooseFolder(): Promise<string | null>;
    };
  }
}
