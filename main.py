from model import Trigram
from optimize import optimize_a

if __name__ == '__main__':
    # Add-a Smoothing
    # Get the init_value(a) which make the perplexity smallest in the val_set
    min_init_value, min_perplexity = optimize_a(img_path='output/perplexity_optimization_en.png')
    # Use the best init_value(a): 'min_init_value' from the validation set.
    model_en = Trigram(init_value=min_init_value, division_ratio=0.9)

    # de
    min_init_value, min_perplexity = optimize_a(img_path='output/perplexity_optimization_de.png')
    model_de = Trigram(init_value=min_init_value, division_ratio=0.9, model_my_path='model/model-de.de',
                       training_path='data/training.de')

    # es
    min_init_value, min_perplexity = optimize_a(img_path='output/perplexity_optimization_es.png')
    model_es = Trigram(init_value=min_init_value, division_ratio=0.9, model_my_path='model/model-es.es',
                       training_path='data/training.es')

    # Train
    model_en.train()
    model_de.train()
    model_es.train()

    # Generation --- 4 Sampling methods
    model_en.generate(sampling_method_num=0)
    # model.generate(sampling_method_num=1, k=4, p=0.005)
    # model.generate(sampling_method_num=2)
    # model.generate(sampling_method_num=3)

    # Test
    model_en.test()
    model_de.test()
    model_es.test()

    # Good-Tuning Smoothing --- If needed, uncomment
    # model = Trigram(init_value=0, division_ratio=0.9)
    # model.train()
    # model.generate(sampling_method_num=0)
    # model_en.test()

    # For Extra Question --- test-port.txt --- If needed, uncomment
    # model_list = ['model/model-en.en', 'model/model-es.es', 'model/model-de.de']
    # for i in range(3):
    #     model = Trigram(init_value=0.1, division_ratio=0.9, model_my_path=model_list[i], test_path='data/test-port')
    #     model.test()