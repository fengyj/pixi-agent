import { Writable } from 'node:stream';
import { appendFileSync } from 'node:fs';

const LEVELS = {
  10: 'trace',
  20: 'debug',
  30: 'info',
  40: 'warn',
  50: 'error',
  60: 'fatal',
};

export default async function memoryTarget(options = {}) {
  const filePath = options.filePath;
  const serviceName = options.serviceName ?? 'pixiagent';
  const serviceVersion = options.serviceVersion ?? 'unknown';
  if (typeof filePath !== 'string' || filePath.length === 0) {
    throw new Error('memory-target requires options.filePath');
  }

  return new Writable({
    write(chunk, _encoding, callback) {
      const raw = typeof chunk === 'string' ? chunk : chunk.toString('utf8');
      for (const line of raw.split('\n')) {
        if (!line.trim()) continue;

        try {
          const record = JSON.parse(line);
          const ecsRecord = {
            ...record,
            '@timestamp': new Date(record.time ?? Date.now()).toISOString(),
            'log.level': LEVELS[record.level] ?? 'info',
            message: record.msg ?? record.message,
            'service.name': record['service.name'] ?? serviceName,
            'service.version': record['service.version'] ?? serviceVersion,
          };
          delete ecsRecord.time;
          delete ecsRecord.level;
          delete ecsRecord.msg;
          appendFileSync(filePath, `${JSON.stringify(ecsRecord)}\n`);
        } catch {
          appendFileSync(filePath, `${line}\n`);
        }
      }
      callback();
    },
  });
}
