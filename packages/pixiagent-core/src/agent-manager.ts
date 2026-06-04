import { ModelProvider, ModelRegistry } from './model';
import { 
    LLMProviderMediaRepository, 
    SessionRepository, 
    WebMediaRepository } from './repository';
import { ToolRegistry } from './tools';

export class AgentManager {
  constructor(public readonly options: AgentManagerOptions) {
    //this.agents = new Map();
  }
}

export interface AgentManagerOptions {
  fileRepo?: {
    webRepo?: WebMediaRepository;
    providerRepos?: Record<ModelProvider, LLMProviderMediaRepository>;
  };
  sessionRepo: SessionRepository;
  models: ModelRegistry;
  tools: ToolRegistry;
}
